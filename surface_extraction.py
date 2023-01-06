import time

import cv2
import cupyx.scipy.ndimage
import numpy as np
import cupy as cupy
import faiss as faiss
from skimage.transform import resize
from skimage.io import imsave
from skimage.filters import threshold_otsu, threshold_multiotsu


class SurfaceExtraction:
	"""
	Standard surface extraction functions.

	Attributes
    ----------
    class_num: int
		Number of pixel class
    cluster_method: string
		Method of clustering z profiles
			'Kmeans' - cluster z frequency profiles
			'Projection' - threshold projections of z frequency profiles
    initial_estimate: string
		Method for initial surface estimation
			'Max' - max intensity positions
	CD: double
		Pixel class weight constant 
	WW: double
		Weight of local smoothness
	THRES: double
		Stopping criteria	

	"""

	class_num = 3
	cluster_method = 'Projection'
	initial_estimate = 'Max'
	CD = 1
	WW = 5
	THRES = 1

	def __init__(self, img, window_size = 1):
		"""
		Initialize an instance for surface extraction.

		Parameters
    	----------
		img: 3D numpy array
			3D image stack
		window_size: int
			Local smoothness window size

		"""
		[self.l, self.h, self.w] = img.shape
		self.img = np.moveaxis(img, 1, 2)
		self.window_size = window_size
    

	def z_profile_fft(self):
		"""
		Fast fourier transformation on z profiles.

		Returns
    	----------
		tempt: 2D numpy array
			Frequency representations of z profiles

		"""
		print("FFT on z profiles...")
		zprof = np.reshape(self.img, [self.l, self.h*self.w])
		
		#  FFT
		tempt = np.fft.fft(zprof, n=self.l, axis=0)
		tempt = np.abs(tempt)

		del_ind = np.arange(np.ceil(self.l/2+1), self.l).tolist()
		del_ind.append(0)
		del_ind = [int(i) for i in del_ind]

		#  Normalization
		tempt = np.delete(tempt, del_ind, axis=0)
		norm_mat = np.tile(np.max(tempt, axis=1)-np.min(tempt, axis=1), [self.w*self.h, 1]).T
		tempt = tempt/norm_mat
		return tempt


	def z_profile_clustering(self, tempt):
		"""
		Kmeans clustering of z frequency profiles.

		Parameters
    	----------
		tempt: 2D numpy array
			Frequency representations of z profiles

		Returns
    	----------
		edge_flag, edge_flag2: 2D numpy array
			Pixel class array

		"""
		print("Clustering z profiles...")
		#  Kmeans clustering
		kmeans = faiss.Kmeans(d=tempt.shape[0], k=self.class_num)
		tempt = np.ascontiguousarray(tempt.T, dtype=np.float32)
		kmeans.train(tempt)
		centers = kmeans.centroids
		labels = kmeans.index.search(x=tempt, k=1)[1].reshape(-1)

		#  Signal, UKnown, Background -> 2, 1, 0
		ind = np.argsort(centers[:,0])
		labels_class = np.zeros(labels.shape)
		centers_class = np.zeros(centers.shape)
		for i in range(self.class_num):
			ind_i = np.where(labels==ind[i])
			labels_class[ind_i] = i
			centers_class[i,:] = centers[ind[i],:]

		edge_flag = np.reshape(labels_class, [self.w, self.h])
		edge_flag2 = (edge_flag)/2
		return edge_flag, edge_flag2


	def z_profile_projection_clustering(self, tempt):
		"""
		Clustering of z frequency profile projections.

		Parameters
    	----------
		tempt: 2D numpy array
			Frequency representations of z profiles

		Returns
    	----------
		edge_flag, edge_flag2: 2D numpy array
			Pixel class array

		"""
		print("Thresholding z profiles...")

		#  Calculate z frequency profiles
		tempt_sum = np.sum(tempt, axis=0)
		tempt_sum = np.reshape(tempt_sum, [self.w, self.h])
		
		#  Multi-thresholding
		#thresholds = threshold_multiotsu(tempt_sum)
		#high_thres = thresholds[0]
		high_thres = threshold_otsu(tempt_sum)
		low_thres = threshold_otsu(tempt_sum[tempt_sum<high_thres])

		#  Assign class label
		label_class = tempt_sum
		label_class -= low_thres
		label_class /= (high_thres - low_thres)
		label_class[label_class <= 0] = 0
		label_class[label_class >= 1] = 1
		#ind_mid_class = np.where(np.logical_and(label_class > 0, label_class < 1))
		#label_class[ind_mid_class] = 0.5

		edge_flag = label_class * 2
		edge_flag2 = edge_flag / 2
		return edge_flag, edge_flag2

  
	def initial_estimate_max_intensity(self):
		"""
		Estimate initial surface with max intensity positions.

		Returns
    	----------
		valm: 2D numpy array
			Max intensity projection
		ind_max: 2D numpy array
			Max intensity positions

		"""
		valm = np.max(self.img, axis=0)
		ind_max = np.argmax(self.img, axis=0)
		return valm, ind_max


	def local_sum(self, img, window_size):
		"""
		Sum of neighboring values.

		Parameters
    	----------
		img: 2D numpy array
			Array of pixel values
		window_size: int
			Size of window within which neighboring values are summed up
			Note a 3 by 3 window (8 neighboring values) has a window_size of 1

		Returns
    	----------
		base_sum: 2D numpy array
			Array containing sum of neighboring values

		"""
		#  Padding
		img_p2 = np.pad(img, window_size, 'symmetric')

		#  Sum
		k = 2 * window_size + 1
		[sz1, sz2] = img_p2.shape
		sz1 = sz1 - 2 * window_size
		sz2 = sz2 - 2 * window_size
		base_sum = np.zeros((sz1, sz2))
		for i in range(k):
			for j in range(k):
				if i == window_size and j == window_size:
					pass
				else:
					base_sum += img_p2[i:sz1+i,j:sz2+j]
		return base_sum


	def local_sum_class(self, img, window_size, sum_pos=0):
		"""
		Calculate neighboring pixel number of a certain class.

		Parameters
    	----------
		img: 2D numpy array
			Array of pixel values
		window_size: int
			Size of window within which pixels are counted
			Note a 3 by 3 window has a window_size of 1
		sum_pos: int
			Pixel class label

		Returns
    	----------
		base_sum: 2D numpy array
			Array containing neighboring pixel number of a certain class

		"""
		#  Padding
		img_p2 = np.pad(img, window_size, 'symmetric')

		#  Count pixel number of the class
		k = 2 * window_size + 1
		[sz1, sz2] = img_p2.shape
		sz1 = sz1 - 2 * window_size
		sz2 = sz2 - 2 * window_size
		base_sum = np.zeros((sz1, sz2))
		for i in range(k):
			for j in range(k):
				sum_temp = np.array([img_p2[i:sz1+i,j:sz2+j]])
				sum_temp = np.sum(sum_temp==sum_pos, axis=0)
				base_sum += sum_temp
		return base_sum


	def local_var(self, img, img_mean, window_size):
		"""
		Variance of neighboring values.

		Parameters
    	----------
		img: 2D numpy array
			Array of pixel values
		img_mean: 2D numpy array
			Array containing mean of neighboring values
		window_size: int
			Size of window within which pixels are counted
			Note a 3 by 3 window (8 neighboring values) has a window_size of 1

		Returns
    	----------
		base_var: 2D numpy array
			Array containing variance of neighboring values

		"""
		#  Padding
		img_p2 = np.pad(img, window_size, 'symmetric')

		#  Variance calculation
		k = 2 * window_size + 1
		[sz1, sz2] = img_p2.shape
		sz1 = sz1 - 2 * window_size
		sz2 = sz2 - 2 * window_size
		base_var = np.zeros((sz1, sz2))
		for i in range(k):
			for j in range(k):
				if i == window_size and j == window_size:
					pass
				else:
					base_var += (img_p2[i:sz1+i,j:sz2+j] - img_mean) * (img_p2[i:sz1+i,j:sz2+j] - img_mean)
		return base_var


	def parameter_setting(self):
		"""
		Parameter initialization for gradient descent-based surface smoothing.

		Returns
    	----------
		lambda1: double
			Initializaion parameter
		edge_flag2: 2D numpy array
			Class labels for each pixel
		ind_max: 2D numpy array
			Initial surface position estimations

		"""
		#  Initialize pixel class labels
		print("Initialization...")
		start = time.time()
		if self.cluster_method == 'Projection':
			tempt = self.z_profile_fft()
			edge_flag, edge_flag2 = self.z_profile_projection_clustering(tempt)
		elif self.cluster_method == 'Kmeans':
			tempt = self.z_profile_fft()
			edge_flag, edge_flag2 = self.z_profile_clustering(tempt)
		else:
			raise ValueError("cluster_method " + self.cluster_method + " doesn't exist!")
		
		#  Estimate initial surface positions
		if self.initial_estimate == 'Max':
			valm, ind_max = self.initial_estimate_max_intensity()
		else:
			raise ValueError("initial_estimate " + self.initial_estimate + " doesn't exist!")
		end = time.time()
		print('Initialization time: {:.2f}'.format(end - start))

		#  Compute histogram for foreground and continous weighted signals
		print("Parameter settings...")
		start = time.time()
		valm_min = np.min(np.min(valm))
		valm_max = np.max(np.max(valm))
		ind_sig = np.where(edge_flag2 == 1)
		ncf, bcf = np.histogram(valm[ind_sig], np.linspace(valm_min, valm_max, 101))
		ncf = ncf/np.sum(ncf)
		ind_uk = np.where(np.logical_and(edge_flag2 > 0, edge_flag2 < 1))
		# ind_uk = np.where(edge_flag2 == 0.5)
		ncb, bcb = np.histogram(valm[ind_uk], np.linspace(valm_min, valm_max, 101))
		ncb = ncb/np.sum(ncb)

		#  Find overlap_2
		nt_ind = np.where(ncb>ncf)
		nt = nt_ind[0][-1]
		bcb_c = (bcb[0:-1] + bcb[1:])/2
		ht = bcb_c[nt]
		sig_ind = np.where(edge_flag2 == 1)
		ind_max_1 = np.where(valm[sig_ind]>ht)
		overlap_2 = (len(sig_ind[0]) - len(ind_max_1[0])) / len(ind_max_1[0])

		#  Calculate neighboring mean and variance of initial surface
		class3 = self.local_sum_class(edge_flag2, self.window_size, sum_pos=1)
		base = self.local_sum(ind_max, self.window_size)
		mold = base/((2 * self.window_size + 1) * (2 * self.window_size + 1) - 1)
		varold2 = self.local_var(ind_max, mold, self.window_size)
		m10 = ind_max - mold
		md = np.zeros(mold.shape)

		#  Calculate sg, dg
		s01 = np.sqrt((varold2+(m10)*(ind_max-(mold+(m10)/9)))/8)
		sD = np.sqrt((varold2+(md)*(mold-(mold+(md)/9)))/8)
		sgain = s01 - sD
		dD = abs(ind_max - mold)
		sg_ind = (class3>8) & (edge_flag2==1)
		sg = sgain[sg_ind]
		dg = dD[sg_ind]
		ind_sg0 = np.where(sg==0)
		sg = np.delete(sg, ind_sg0, axis=0)
		dg = np.delete(dg, ind_sg0, axis=0)

		#  Adjust overlap_2
		if overlap_2 < 0:
			overlap_2 = 0
		elif overlap_2 > 0.5:
			overlap_2 = 0.5

		#  Find lambda1
		WA = dg/sg
		lambda1 = np.abs(np.quantile(WA, overlap_2))
		end = time.time()
		print('Parameter settings time: {:.2f}'.format(end - start))
		return lambda1, edge_flag2, ind_max


	def surface_smooth(self, lambda1, edge_flag2, ind_max):
		"""
		Gradient-descent-based smooth manifold extraction.

		Parameters
    	----------
		lambda1: double
			Initializaion parameter
		edge_flag2: 2D numpy array
			Class labels for each pixel
		ind_max: 2D numpy array
			Initial surface position estimations
		
		Returns
    	----------
		ind_maxk: 2D numpy array
			Extracted surface positions

		"""
		#  Calculate pixel weights based on class labels
		print("Smoothing surface...")
		start = time.time()
		edge1 = np.where(edge_flag2==1)
		#edge5 = np.where(edge_flag2==0.5)
		edge5 = np.where(np.logical_and(edge_flag2 > 0, edge_flag2 < 1))
		edge0 = np.where(edge_flag2==0)

		npxl1 = len(edge1[0])
		npxl5 = len(edge5[0])
		npxl0 = len(edge0[0])

		c1=self.CD*1/lambda1
		c2=self.CD*1/lambda1
		c3=self.CD*0/lambda1
		edge_flag3 = np.zeros(edge_flag2.shape)

		edge_flag3[edge1]=c1  
		edge_flag3[edge5]=c2 * edge_flag2[edge5]
		edge_flag3[edge0]=c3

		#  Calculate step size
		KE = np.max(ind_max[edge_flag2>0]) - np.min(ind_max[edge_flag2>0]) + 1 
		step = KE/100

		#  Downsample ratios for pyramidal index map optimization
		npxl = self.h*self.w
		if np.sqrt(npxl)<= 512:   
			ratio = np.array([1])
		elif np.sqrt(npxl) > 512 and np.sqrt(npxl) <= 1024:
			ratio = np.array([0.5, 1])
		elif np.sqrt(npxl) > 1024 and np.sqrt(npxl) <= 2048:
			ratio = np.array([0.25, 0.5, 1])
		elif np.sqrt(npxl) > 2048 and np.sqrt(npxl) <= 4096:
			ratio = np.array([0.125, 0.25, 0.5, 1])
		elif np.sqrt(npxl) > 4096 and np.sqrt(npxl) <= 8192:
			ratio = np.array([0.0625, 0.125, 0.25, 0.5, 1])
		elif np.sqrt(npxl) > 8192:
			ratio = np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1])

		#  Surface positions
		ind_maxk = np.float32(ind_max)
		#costA = np.array([])

		#  Pixel class map
		edge1_map = np.zeros(edge_flag2.shape)
		edge1_map[edge1] = 1
		edge5_map = np.zeros(edge_flag2.shape)
		edge5_map[edge5] = 1
		edge0_map = np.zeros(edge_flag2.shape)
		edge0_map[edge0] = 1

		#  Position updates at each iteration
		shiftc = np.zeros(edge_flag2.shape)

		#  Pyramidal index map optimization
		for rn in range(len(ratio)):
			#  Resize to different scales
			sratio = np.array([np.round(ratio[rn]*self.h), np.round(ratio[rn]*self.w)])
			sratio = sratio.astype('int')
			srn = ratio[rn] * ratio[rn]

			ind_maxk = cv2.resize(ind_maxk,(sratio[0],sratio[1]),interpolation=cv2.INTER_NEAREST) # different from matlab
			ind_maxk2 = cv2.resize(ind_max,(sratio[0],sratio[1]),interpolation=cv2.INTER_NEAREST)
			edge_flag3_2 = cv2.resize(edge_flag3,(sratio[0],sratio[1]),interpolation=cv2.INTER_NEAREST)
			edge1_2 = cv2.resize(edge1_map,(sratio[0],sratio[1]),interpolation=cv2.INTER_NEAREST)
			edge5_2 = cv2.resize(edge5_map,(sratio[0],sratio[1]),interpolation=cv2.INTER_NEAREST)
			edge0_2 = cv2.resize(edge0_map,(sratio[0],sratio[1]),interpolation=cv2.INTER_NEAREST)
			shiftc2 = cv2.resize(shiftc,(sratio[0],sratio[1]),interpolation=cv2.INTER_NEAREST)

			#  Energy function values
			cost = np.array([])
			cost = np.append(cost, 100)
			cost = np.append(cost, 10) #fake value to enter the loop

			itern = 1
			lim = 1/ratio[rn]

			#  Stopping criteria
			while np.abs(cost[itern]) > 0.001*KE*lim*self.THRES:
				itern += 1
				ind_max1 = ind_maxk + step
				ind_max2 = ind_maxk - step

				#  Local smoothness
				base = self.local_sum(ind_maxk, self.window_size)
				mold = base/((2 * self.window_size + 1) * (2 * self.window_size + 1) - 1)
				varold2 = self.local_var(ind_maxk, mold, self.window_size)

				#  Gradient of the energy function
				d1 = np.abs(ind_maxk2 - ind_max1) * edge_flag3_2
				d2 = np.abs(ind_maxk2 - ind_max2) * edge_flag3_2

				m11 = ind_max1 - mold
				m12 = ind_max2 - mold

				s1 = self.WW * np.sqrt((varold2+(m11)*(ind_max1-(mold+(m11)/9)))/8)
				s2 = self.WW * np.sqrt((varold2+(m12)*(ind_max2-(mold+(m12)/9)))/8)

				c1 = d1 + s1
				c2 = d2 + s2
				dt = c1 - c2

				#  Add momentum
				shiftc2 = 0.5 * shiftc2 + 50 * dt * step

				#  Update surface positions
				ind_maxk = ind_maxk - shiftc2

				#  Updated energy function value
				ind_edge1_2 = np.where(edge1_2 == 1)
				ind_edge5_2 = np.where(edge5_2 == 1)
				ind_edge0_2 = np.where(edge0_2 == 1)
				cost_val = np.sum(np.abs(dt[ind_edge1_2]))/(srn*npxl1) + np.sum(np.abs(dt[ind_edge5_2]))/(srn*npxl5) + np.sum(np.abs(dt[ind_edge0_2]))/(srn*npxl0)
				cost = np.append(cost, cost_val)
				
				step = step * 0.99
		end = time.time()
		print('Surface smoothing time: {:.2f}'.format(end - start))
		return ind_maxk


class RapidSurfaceExtraction(SurfaceExtraction):
	"""
	Accelerated surface extraction functions.
	Inheritance of SurfaceExtraction.

	"""

	def rapid_local_sum(self, img, window_size):
		"""
		Sum of neighboring values - parallel computing.

		Parameters
    	----------
		img: 2D numpy array
			Array of pixel values
		window_size: int
			Size of window within which neighboring values are summed up
			Note a 3 by 3 window (8 neighboring values) has a window_size of 1

		Returns
    	----------
		base_sum: 2D numpy array
			Array containing sum of neighboring values

		"""
		#  Padding
		img_p2 = cupy.pad(img, window_size, 'symmetric')

		#  Sum of neighboring values
		k = 2 * window_size + 1
		[sz1, sz2] = img_p2.shape
		sz1 = sz1 - 2 * window_size
		sz2 = sz2 - 2 * window_size
		base_sum = cupy.zeros((sz1, sz2))
		for i in range(k):
			for j in range(k):
				if i == window_size and j == window_size:
					pass
				else:
					base_sum += img_p2[i:sz1+i,j:sz2+j]
		return base_sum


	def rapid_local_sum_class(self, img, window_size, sum_pos=0):
		"""
		Calculate neighboring pixel number of a certain class - parallel computing.

		Parameters
    	----------
		img: 2D numpy array
			Array of pixel values
		window_size: int
			Size of window within which pixels are counted
			Note a 3 by 3 window has a window_size of 1
		sum_pos: int
			Pixel class label

		Returns
    	----------
		base_sum: 2D numpy array
			Array containing neighboring pixel number of a certain class

		"""
		#  Padding
		img_p2 = cupy.pad(img, window_size, 'symmetric')

		#  Count pixel number of the class
		k = 2 * window_size + 1
		[sz1, sz2] = img_p2.shape
		sz1 = sz1 - 2 * window_size
		sz2 = sz2 - 2 * window_size
		base_sum = cupy.zeros((sz1, sz2))
		for i in range(k):
			for j in range(k):
				sum_temp = cupy.array([img_p2[i:sz1+i,j:sz2+j]])
				sum_temp = cupy.sum(sum_temp==sum_pos, axis=0)
				base_sum += sum_temp
		return base_sum


	def rapid_local_var(self, img, img_mean, window_size):
		"""
		Variance of neighboring values - parallel computing.

		Parameters
    	----------
		img: 2D numpy array
			Array of pixel values
		img_mean: 2D numpy array
			Array containing mean of neighboring values
		window_size: int
			Size of window within which pixels are counted
			Note a 3 by 3 window (8 neighboring values) has a window_size of 1

		Returns
    	----------
		base_var: 2D numpy array
			Array containing variance of neighboring values

		"""
		#  Padding
		img_p2 = cupy.pad(img, window_size, 'symmetric')

		#  Variance calculation
		k = 2 * window_size + 1
		[sz1, sz2] = img_p2.shape
		sz1 = sz1 - 2 * window_size
		sz2 = sz2 - 2 * window_size
		base_var = cupy.zeros((sz1, sz2))
		for i in range(k):
			for j in range(k):
				if i == window_size and j == window_size:
					pass
				else:
					base_var += (img_p2[i:sz1+i,j:sz2+j] - img_mean) * (img_p2[i:sz1+i,j:sz2+j] - img_mean)
		return base_var


	def rapid_parameter_setting(self):
		"""
		Rapid parameter initialization for gradient descent-based surface smoothing.
		Parallel computing.

		Returns
    	----------
		lambda1: double
			Initializaion parameter
		edge_flag2: 2D numpy array
			Class labels for each pixel
		ind_max: 2D numpy array
			Initial surface position estimations

		"""
		print("Initialization...")
		#  Initialize pixel class labels
		start = time.time()
		if self.cluster_method == 'Projection':
			tempt = self.z_profile_fft()
			edge_flag, edge_flag2 = self.z_profile_projection_clustering(tempt)
		elif self.cluster_method == 'Kmeans':
			tempt = self.z_profile_fft()
			edge_flag, edge_flag2 = self.z_profile_clustering(tempt)
		else:
			raise ValueError("cluster_method " + self.cluster_method + " doesn't exist!")

		#  Estimate initial surface positions
		if self.initial_estimate == 'Max':
			valm, ind_max = self.initial_estimate_max_intensity()
		else:
			raise ValueError("initial_estimate " + self.initial_estimate + " doesn't exist!")
		end = time.time()
		print('Initialization time: {:.2f}'.format(end - start))

		#  Memory management
		print("Parameter settings...")
		start = time.time()
		mempool = cupy.get_default_memory_pool()
		pinned_mempool = cupy.get_default_pinned_memory_pool()
		valm = cupy.array(valm, dtype='float32')
		ind_max = cupy.array(ind_max, dtype='float32')
		edge_flag2 = cupy.array(edge_flag2, dtype='float32')
		
		#  Compute histogram for foreground and continous weighted signals
		valm_min = cupy.min(cupy.min(valm))
		valm_max = cupy.max(cupy.max(valm))
		ind_sig = cupy.where(edge_flag2 == 1)
		ncf, bcf = cupy.histogram(valm[ind_sig], cupy.linspace(valm_min, valm_max, 101))
		ncf = ncf/cupy.sum(ncf)
		ind_uk = cupy.where(cupy.logical_and(edge_flag2 > 0, edge_flag2 < 1))
		# ind_uk = np.where(edge_flag2 == 0.5)
		ncb, bcb = cupy.histogram(valm[ind_uk], np.linspace(valm_min, valm_max, 101))
		ncb = ncb/cupy.sum(ncb)

		#  Find overlap_2
		nt_ind = cupy.where(ncb>ncf)
		nt = nt_ind[0][-1]
		bcb_c = (bcb[0:-1] + bcb[1:])/2
		ht = bcb_c[nt]
		sig_ind = cupy.where(edge_flag2 == 1)
		ind_max_1 = cupy.where(valm[sig_ind]>ht)
		overlap_2 = (len(sig_ind[0]) - len(ind_max_1[0])) / len(ind_max_1[0])

		#  Calculate neighboring mean and variance of initial surface
		class3 = self.rapid_local_sum_class(edge_flag2, self.window_size, sum_pos=1)
		base = self.rapid_local_sum(ind_max, self.window_size)
		mold = base/((2 * self.window_size + 1) * (2 * self.window_size + 1) - 1)
		varold2 = self.rapid_local_var(ind_max, mold, self.window_size)
		m10 = ind_max - mold
		md = cupy.zeros(mold.shape)

		#  Calculate sg, dg
		s01 = cupy.sqrt((varold2+(m10)*(ind_max-(mold+(m10)/9)))/8)
		sD = cupy.sqrt((varold2+(md)*(mold-(mold+(md)/9)))/8)
		sgain=s01-sD
		dD=cupy.abs(ind_max-mold)
		sg_ind = (class3>8) & (edge_flag2==1)
		sg=sgain[sg_ind]
		dg=dD[sg_ind]
		ind_sg0 = cupy.where(sg==0)
		mask_sg = cupy.ones(len(sg), dtype=bool)
		mask_sg[ind_sg0] = False
		sg = sg[mask_sg]
		mask_dg = cupy.ones(len(dg), dtype=bool)
		mask_dg[ind_sg0] = False
		dg = dg[mask_dg]

		#  Adjust overlap_2
		if overlap_2 < 0:
			overlap_2 = 0
		elif overlap_2 > 0.5:
			overlap_2 = 0.5

		#  Find lambda1
		WA = dg/sg
		WA = cupy.asnumpy(WA)
		lambda1 = np.abs(np.quantile(WA,overlap_2))
		edge_flag2 = cupy.asnumpy(edge_flag2)
		ind_max = cupy.asnumpy(ind_max)
		mempool.free_all_blocks()
		pinned_mempool.free_all_blocks()
		end = time.time()
		print('Parameter settings time: {:.2f}'.format(end - start))
		return lambda1, edge_flag2, ind_max


	def surface_smooth(self, lambda1, edge_flag2, ind_max):
		"""
		Gradient-descent-based smooth manifold extraction.

		Parameters
    	----------
		lambda1: double
			Initializaion parameter
		edge_flag2: 2D numpy array
			Class labels for each pixel
		ind_max: 2D numpy array
			Initial surface position estimations
		
		Returns
    	----------
		ind_maxk: 2D numpy array
			Extracted surface positions

		"""
		#  Calculate pixel weights based on class labels
		print("Smoothing surface...")
		start = time.time()
		edge1 = np.where(edge_flag2==1)
		# edge5 = np.where(edge_flag2==0.5)
		edge5 = np.where(np.logical_and(edge_flag2 > 0, edge_flag2 < 1))
		edge0 = np.where(edge_flag2==0)

		npxl1 = len(edge1[0])
		npxl5 = len(edge5[0])
		npxl0 = len(edge0[0])

		c1=self.CD*1/lambda1
		c2=self.CD*1/lambda1
		c3=self.CD*0/lambda1
		edge_flag3 = np.zeros(edge_flag2.shape)

		edge_flag3[edge1]=c1  
		edge_flag3[edge5]=c2 * edge_flag2[edge5]
		edge_flag3[edge0]=c3

		#  Calculate step size
		KE = np.max(ind_max[edge_flag2>0]) - np.min(ind_max[edge_flag2>0]) + 1 
		step = KE/100

		#  Downsample ratios for pyramidal index map optimization
		npxl = self.h*self.w
		if np.sqrt(npxl)<= 512:   
			ratio = np.array([1])
		elif np.sqrt(npxl) > 512 and np.sqrt(npxl) <= 1024:
			ratio = np.array([0.5, 1])
		elif np.sqrt(npxl) > 1024 and np.sqrt(npxl) <= 2048:
			ratio = np.array([0.25, 0.5, 1])
		elif np.sqrt(npxl) > 2048 and np.sqrt(npxl) <= 4096:
			ratio = np.array([0.125, 0.25, 0.5, 1])
		elif np.sqrt(npxl) > 4096 and np.sqrt(npxl) <= 8192:
			ratio = np.array([0.0625, 0.125, 0.25, 0.5, 1])
		elif np.sqrt(npxl) > 8192:
			ratio = np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1])

		#  Surface positions
		ind_max = cupy.array(ind_max, dtype='float32')
		ind_maxk = cupy.array(ind_max, dtype='float32')

		#  Pixel class map
		edge_flag3 = cupy.array(edge_flag3, dtype='float32')
		edge1_map = cupy.zeros(edge_flag2.shape, dtype='float32')
		edge1_map[edge1] = 1
		edge5_map = cupy.zeros(edge_flag2.shape, dtype='float32')
		edge5_map[edge5] = 1
		edge0_map = cupy.zeros(edge_flag2.shape, dtype='float32')
		edge0_map[edge0] = 1

		#  Position updates at each iteration
		shiftc=cupy.zeros(edge_flag2.shape, dtype='float32')

		# Memory management
		mempool = cupy.get_default_memory_pool()
		pinned_mempool = cupy.get_default_pinned_memory_pool()

		#  Pyramidal index map optimization
		for rn in range(len(ratio)):
			#  Resize to different scales
			ind_maxk2 = cupyx.scipy.ndimage.zoom(ind_max, ratio[rn], order=0)
			edge_flag3_2 = cupyx.scipy.ndimage.zoom(edge_flag3, ratio[rn], order=0)
			edge1_2 = cupyx.scipy.ndimage.zoom(edge1_map, ratio[rn], order=0)
			edge5_2 = cupyx.scipy.ndimage.zoom(edge5_map, ratio[rn], order=0)
			edge0_2 = cupyx.scipy.ndimage.zoom(edge0_map, ratio[rn], order=0)
			shiftc2 = cupyx.scipy.ndimage.zoom(shiftc, ratio[rn], order=0)
			if rn == 0:
				zoom_scale = [ratio[0], ratio[0]]
			else:
				zoom_scale = [ind_maxk2.shape[0]/ind_maxk.shape[0], ind_maxk2.shape[1]/ind_maxk.shape[1]]
			ind_maxk = cupyx.scipy.ndimage.zoom(ind_maxk, zoom_scale, order=0)

			#  Energy function values
			srn = ratio[rn] * ratio[rn]
			cost = np.array([])
			cost = np.append(cost, 100)
			cost = np.append(cost, 10) #fake value to enter the loop
			
			itern = 1
			lim = 1/ratio[rn]

			#  Stopping criteria
			while np.abs(cost[itern])>0.0001*KE*lim*self.THRES:
				itern += 1
				ind_max1 = ind_maxk + step
				ind_max2 = ind_maxk - step

				#  Local smoothness
				base = self.rapid_local_sum(ind_maxk, self.window_size)
				mold = base/((2 * self.window_size + 1) * (2 * self.window_size + 1) - 1)
				varold2 = self.rapid_local_var(ind_maxk, mold, self.window_size)

				#  Gradient of the energy function
				d1 = cupy.abs(ind_maxk2 - ind_max1) * edge_flag3_2
				d2 = cupy.abs(ind_maxk2 - ind_max2) * edge_flag3_2

				m11 = ind_max1 - mold
				m12 = ind_max2 - mold

				s1 = self.WW * cupy.sqrt((varold2+(m11)*(ind_max1-(mold+(m11)/9)))/8)
				s2 = self.WW * cupy.sqrt((varold2+(m12)*(ind_max2-(mold+(m12)/9)))/8)

				c1 = d1 + s1
				c2 = d2 + s2
				dt = c1 - c2

				#  Add momentum
				shiftc2 = 0.5 * shiftc2 + 50 * dt * step

				#  Update surface positions
				ind_maxk = ind_maxk - shiftc2

				#  Updated energy function value
				ind_edge1_2 = cupy.where(edge1_2 == 1)
				ind_edge5_2 = cupy.where(edge5_2 == 1)
				ind_edge0_2 = cupy.where(edge0_2 == 1)
				cost_val = cupy.sum(cupy.abs(dt[ind_edge1_2]))/(srn*npxl1) + cupy.sum(cupy.abs(dt[ind_edge5_2]))/(srn*npxl5) + cupy.sum(cupy.abs(dt[ind_edge0_2]))/(srn*npxl0)
				cost_val = cupy.asnumpy(cost_val)
				cost = np.append(cost, cost_val)
				step = step * 0.99

				# print('Itration inner ' + str(itern))
				# mempool.free_all_blocks()
				# pinned_mempool.free_all_blocks()
				# print(mempool.used_bytes())
				# print(mempool.total_bytes()) 	  	

			#  Memory management  
			print('Itration ' + str(rn))
			mempool.free_all_blocks()
			pinned_mempool.free_all_blocks() 
			print("Memory in use: {:}".format(mempool.used_bytes()))
			print("Memory in total: {:}".format(mempool.total_bytes()))

		ind_maxk = cupy.asnumpy(ind_maxk)
		end = time.time()
		print('Surface smoothing time: {:.2f}'.format(end - start))
		return ind_maxk