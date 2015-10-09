## @package rot_utils
#  Util functions for dealing with rotations
#

import scipy.io as sio
import numpy as np
from   scipy import linalg as linalg
import sys, os
import pdb
import math
from mpl_toolkits.mplot3d import Axes3D

_FLOAT_EPS_4 = np.finfo(float).eps * 4.0

def get_rot_angle(view1, view2):
	try:
		viewDiff = linalg.logm(np.dot(view2, np.transpose(view1)))
	except:
		print "Error Encountered"
		pdb.set_trace()

	viewDiff = linalg.norm(viewDiff, ord='fro')
	assert not any(np.isnan(viewDiff.flatten()))
	assert not any(np.isinf(viewDiff.flatten()))
	angle    = viewDiff/np.sqrt(2)
	return angle


def get_cluster_assignments(x, centers):
	N       = x.shape[0]
	nCl     = centers.shape[0]	
	distMat = np.inf * np.ones((nCl,N))

	for c in range(nCl):
		for i in range(N):
			distMat[c,i] = get_rot_angle(centers[c], x[i])

	assert not any(np.isinf(distMat.flatten()))
	assert not any(np.isnan(distMat.flatten()))
	
	assgn    = np.argmin(distMat, axis=0)
	minDist  = np.amin(distMat, axis=0)
	meanDist = np.mean(minDist) 
	assert all(minDist.flatten()>=0)
	return assgn, meanDist
 

def karcher_mean(x, tol=0.01):
	'''
	Determined the Karcher mean of rotations
	Implementation from Algorithm 1, Rotation Averaging, Hartley et al, IJCV 2013
	'''
	R = x[0]
	N = x.shape[0]
	normDeltaR = np.inf
	itr = 0
	while True:
		#Estimate the delta rotation between the current center and all points
		deltaR  = np.zeros((3,3))
		oldNorm = normDeltaR
		for i in range(N):
			deltaR += linalg.logm(np.dot(np.transpose(R),x[i]))
		deltaR     = deltaR / N
		normDeltaR = linalg.norm(deltaR, ord='fro')/np.sqrt(2)

		if oldNorm - normDeltaR < tol:
			break
	
		R = np.dot(R, linalg.expm(deltaR)) 
		#print itr
		itr += 1		
	
	return R
	

def estimate_clusters(x, assgn, nCl):
	clusters = np.zeros((nCl,3,3))
	for c in range(nCl):
		pointSet    = x[assgn==c]
		clusters[c] = karcher_mean(pointSet) 	

	return clusters	
	

def cluster_rotmats(x,nCl=2,tol=0.01):
	'''
	x  : numMats * 3 * 3
	nCl: number of clusters
	tol: tolerance when to stop, it is basically if the reduction in mean error goes below this point 
	'''
	assert x.shape[1]==x.shape[2]==3
	N  = x.shape[0]

	#Randomly chose some points as initial cluster centers
	perm        = np.random.permutation(N)
	centers     = x[perm[0:nCl]] 
	assgn, dist = get_cluster_assignments(x, centers)	
	print "Initial Mean Distance is: %f" % dist

	itr = 0
	clusterFlag = True
	while clusterFlag:
		itr        += 1
		prevAssgn  = np.copy(assgn)
		prevDist   = dist
		#Find the new centers
		centers    = estimate_clusters(x, assgn, nCl)
		#Find the new assgn
		assgn,dist = get_cluster_assignments(x, centers)

		print "iteration: %d, mean distance: %f" % (itr,dist)

		if prevDist - dist < tol:
			print "Desired tolerance achieved"
			clusterFlag = False

		if all(assgn==prevAssgn):
			print "Assignments didnot change in this iteration, hence converged"
			clusterFlag = False

	return assgn, centers	 	


def axis_to_skewsym(v):
	'''
		Converts an axis into a skew symmetric matrix format. 
	'''
	v = v/np.linalg.norm(v)
	vHat = np.zeros((3,3))
	vHat[0,1], vHat[0,2] = -v[2],v[1]
	vHat[1,0], vHat[1,2] = v[2],-v[0]
	vHat[2,0], vHat[2,1] = -v[1],v[0] 

	return vHat


def angle_axis_to_rotmat(theta, v):
	'''
		Given the axis v, and a rotation theta - convert it into rotation matrix
		theta needs to be in radian
	'''	
	assert theta>=0 and theta<np.pi, "Invalid theta"

	vHat   = axis_to_skewsym(v)
	vHatSq = np.dot(vHat, vHat)
	#Rodrigues Formula
	rotMat = np.eye(3) + math.sin(theta) * vHat + (1 - math.cos(theta)) * vHatSq
	return rotMat
	 

def rotmat_to_angle_axis(rotMat):
	'''
		Converts a rotation matrix into angle axis format
	'''
	aa = linalg.logm(rotMat)
	aa = (aa - aa.transpose()	)/2.0
	v1,v2,v3 = -aa[1,2], aa[0,2], -aa[0,1]
	v  = np.array((v1,v2,v3))
	theta = np.linalg.norm(v)
	if theta>0:
		v     = v/theta
	return theta, v

##
# Convert Euler matrices into a rotation matrix. 
def euler2mat(z=0, y=0, x=0, isRadian=True):
	''' Return matrix for rotations around z, y and x axes

	Uses the z, then y, then x convention above

	Parameters
	----------
	z : scalar
		 Rotation angle in radians around z-axis (performed first)
	y : scalar
		 Rotation angle in radians around y-axis
	x : scalar
		 Rotation angle in radians around x-axis (performed last)

	Returns
	-------
	M : array shape (3,3)
		 Rotation matrix giving same rotation as for given angles

	Examples
	--------
	>>> zrot = 1.3 # radians
	>>> yrot = -0.1
	>>> xrot = 0.2
	>>> M = euler2mat(zrot, yrot, xrot)
	>>> M.shape == (3, 3)
	True

	The output rotation matrix is equal to the composition of the
	individual rotations

	>>> M1 = euler2mat(zrot)
	>>> M2 = euler2mat(0, yrot)
	>>> M3 = euler2mat(0, 0, xrot)
	>>> composed_M = np.dot(M3, np.dot(M2, M1))
	>>> np.allclose(M, composed_M)
	True

	You can specify rotations by named arguments

	>>> np.all(M3 == euler2mat(x=xrot))
	True

	When applying M to a vector, the vector should column vector to the
	right of M.  If the right hand side is a 2D array rather than a
	vector, then each column of the 2D array represents a vector.

	>>> vec = np.array([1, 0, 0]).reshape((3,1))
	>>> v2 = np.dot(M, vec)
	>>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
	>>> vecs2 = np.dot(M, vecs)

	Rotations are counter-clockwise.

	>>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
	>>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
	True
	>>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
	>>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
	True
	>>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
	>>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
	True

	Notes
	-----
	The direction of rotation is given by the right-hand rule (orient
	the thumb of the right hand along the axis around which the rotation
	occurs, with the end of the thumb at the positive end of the axis;
	curl your fingers; the direction your fingers curl is the direction
	of rotation).  Therefore, the rotations are counterclockwise if
	looking along the axis of rotation from positive to negative.
	'''

	if not isRadian:
		z = ((np.pi)/180.) * z
		y = ((np.pi)/180.) * y
		x = ((np.pi)/180.) * x
	assert z>=(-np.pi) and z < np.pi, 'Inapprorpriate z: %f' % z
	assert y>=(-np.pi) and y < np.pi, 'Inapprorpriate y: %f' % y
	assert x>=(-np.pi) and x < np.pi, 'Inapprorpriate x: %f' % x	

	Ms = []
	if z:
			cosz = math.cos(z)
			sinz = math.sin(z)
			Ms.append(np.array(
							[[cosz, -sinz, 0],
							 [sinz, cosz, 0],
							 [0, 0, 1]]))
	if y:
			cosy = math.cos(y)
			siny = math.sin(y)
			Ms.append(np.array(
							[[cosy, 0, siny],
							 [0, 1, 0],
							 [-siny, 0, cosy]]))
	if x:
			cosx = math.cos(x)
			sinx = math.sin(x)
			Ms.append(np.array(
							[[1, 0, 0],
							 [0, cosx, -sinx],
							 [0, sinx, cosx]]))
	if Ms:
			return reduce(np.dot, Ms[::-1])
	return np.eye(3)


def mat2euler(M, cy_thresh=None, seq='zyx'):
	''' 
	Taken Forom: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
	Discover Euler angle vector from 3x3 matrix

	Uses the conventions above.

	Parameters
	----------
	M : array-like, shape (3,3)
	cy_thresh : None or scalar, optional
		 threshold below which to give up on straightforward arctan for
		 estimating x rotation.  If None (default), estimate from
		 precision of input.

	Returns
	-------
	z : scalar
	y : scalar
	x : scalar
		 Rotations in radians around z, y, x axes, respectively

	Notes
	-----
	If there was no numerical error, the routine could be derived using
	Sympy expression for z then y then x rotation matrix, which is::

		[                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
		[cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
		[sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

	with the obvious derivations for z, y, and x

		 z = atan2(-r12, r11)
		 y = asin(r13)
		 x = atan2(-r23, r33)

	for x,y,z order
		y = asin(-r31)
		x = atan2(r32, r33)
    z = atan2(r21, r11)


	Problems arise when cos(y) is close to zero, because both of::

		 z = atan2(cos(y)*sin(z), cos(y)*cos(z))
		 x = atan2(cos(y)*sin(x), cos(x)*cos(y))

	will be close to atan2(0, 0), and highly unstable.

	The ``cy`` fix for numerical instability below is from: *Graphics
	Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
	0123361559.  Specifically it comes from EulerAngles.c by Ken
	Shoemake, and deals with the case where cos(y) is close to zero:

	See: http://www.graphicsgems.org/

	The code appears to be licensed (from the website) as "can be used
	without restrictions".
	'''
	M = np.asarray(M)
	if cy_thresh is None:
			try:
					cy_thresh = np.finfo(M.dtype).eps * 4
			except ValueError:
					cy_thresh = _FLOAT_EPS_4
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
	# cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
	cy = math.sqrt(r33*r33 + r23*r23)
	if seq=='zyx':
		if cy > cy_thresh: # cos(y) not close to zero, standard form
				z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
				y = math.atan2(r13,  cy) # atan2(sin(y), cy)
				x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
		else: # cos(y) (close to) zero, so x -> 0.0 (see above)
				# so r21 -> sin(z), r22 -> cos(z) and
				z = math.atan2(r21,  r22)
				y = math.atan2(r13,  cy) # atan2(sin(y), cy)
				x = 0.0
	elif seq=='xyz':
		if cy > cy_thresh:
			y = math.atan2(-r31, cy)
			x = math.atan2(r32, r33)
			z = math.atan2(r21, r11)
		else:
		  z = 0.0
			if r31 < 0:
				y = np.pi/2
				x = atan2(r12, r13)	
			else:
				y = -np.pi/2
				#x = 
	else:
		raise Exception('Sequence not recognized')
	return z, y, x


def euler2quat(z=0, y=0, x=0):
    ''' Return quaternion corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Notes
    -----
    We can derive this formula in Sympy using:

    1. Formula giving quaternion corresponding to rotation of theta radians
       about arbitrary axis:
       http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
       theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
       http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
       formulae from 2.) to give formula for combined rotations.
    '''
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])


def quat2euler(q):
    ''' Return Euler angles corresponding to quaternion `q`

    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion

    Returns
    -------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``quat2mat`` and ``mat2euler`` functions, but
    the reduction in computation is small, and the code repetition is
    large.
    '''
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    return mat2euler(nq.quat2mat(q))

def plot_rotmats(rotMats, isInteractive=True):
	if isInteractive:
		import matplotlib
		matplotlib.use('tkagg')
		import matplotlib.pyplot as plt
	else:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
	
	N = rotMats.shape[0]
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	xpos, ypos, zpos = np.zeros((N,1)), np.zeros((N,1)), np.zeros((N,1))
	vx,vy,vz = [],[],[]

	for i in range(N):
		theta,v = rotmat_to_angle_axis(rotMats[i])
		v       = theta * v
		vx.append(v[0])
		vy.append(v[1])
		vz.append(v[2])

	ax.quiver(xpos,ypos,zpos,vx,vy,vz)
	plt.show()	
	ax.set_xlim(-1,1)
	ax.set_ylim(-1,1)
	ax.set_zlim(-1,1)


def generate_random_rotmats(numMat = 100, thetaRange=np.pi/4, thetaFixed=False):
	rotMats = np.zeros((numMat,3,3))

	if not thetaFixed:
		#Randomly generate an axis for rotation matrix
		v    = np.random.random(3)
		for i in range(numMat):
			theta      = thetaRange * np.random.random()					
			rotMats[i] = angle_axis_to_rotmat(theta, v)
	else:
		for i in range(numMat):
			v    = np.random.randn(3)
			v    = v/linalg.norm(v)
			theta      = thetaRange * np.random.random()					
			rotMats[i] = angle_axis_to_rotmat(theta, v)
		
	return rotMats


def test_clustering():
	'''
	For testing clustering:
	Randomly generate soem data, cluster it and save it .mat file
	Using matlab I will then visualize it. Visualizing in python is being a pain. 
	'''
	N   = 1000
	nCl = 3

	#Generate the data using nCl different axes. 
	dat = np.zeros((N,3,3))
	idx = np.linspace(0,N,nCl+1).astype('int')
	for i in range(nCl):
		dat[idx[i]:idx[i+1]] = generate_random_rotmats(idx[i+1]-idx[i],thetaFixed=True)	

	assgn, centersMat = cluster_rotmats(dat,nCl)

	points = np.zeros((N,3))
	for i in range(N):
		theta,points[i] = rotmat_to_angle_axis(dat[i])
		points[i] = theta*points[i]

	centers = np.zeros((nCl,3))
	for i in range(nCl):
		theta,centers[i] = rotmat_to_angle_axis(centersMat[i])
		centers[i] = theta*centers[i]

	sio.savemat('test_clustering.mat',{'assgn':assgn,'centers':centers,'points':points})	 
	
	
