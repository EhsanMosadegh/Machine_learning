import numpy as np
import numpy.linalg as linalg
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import sklearn as sk
#from sklearn.preprocessing import MinMaxScaler


def compute_Z_col(X, centering=True, scaling=False):
    X = X.transpose()
    if(centering):
        m = X.mean(axis=0) # mean of features==row
        X = X - m # ???
    if(scaling):
        X = X/np.std(X,axis=0) # std of features==row
    return X.transpose()

def compute_Z(X, centering=True, scaling=False):

    if(centering):
        m = X.mean(axis=0) # mean of features==row
        X = X - m # ???
    if(scaling):
        X = X/np.std(X,axis=0) # std of features==row
    return X

def compute_covariance_matrix(Z):
    cov_matrix = np.cov(Z.transpose())
    return cov_matrix

def find_pcs(COV):
    eigenValues, eigenVectors = linalg.eig(COV)
    idx = eigenValues.argsort()[::-1] # how sorts?

    L = eigenValues[idx] # eigenValue
    PCS = eigenVectors[:, idx] # eigenVector / is it from largest->smallest?

    return PCS, L

def project_data(Z, PCS, L, k, var):
    if(k>0):
        k_PCS = PCS[:,:k] # ???
        Z_star = Z.dot(k_PCS)
        return Z_star
    else:
        sum_L = np.sum(L)
        for i in range(len(L)):
            sum_var = np.sum(L[:i+1])
            if(sum_var/sum_L >=  var):
                k = i
                break
        # print("K is : ", k+1 )
        k_PCS = PCS[:,:k].transpose()
        Z_star = Z.dot(k_PCS)
        return Z_star

def compress_images(DATA,k):
    Z = compute_Z_col(DATA,True,True)
    cov_matrix = compute_covariance_matrix(Z)
    PCS,L = find_pcs(cov_matrix)
    U = PCS[:,:k] # ???
    Z_star = project_data(Z, PCS, L, k, 0)
    U_transpose = U.transpose()
    X_compressed = np.dot(Z_star,U_transpose)
    #X_compressed = X_compressed/255

    # im = Image.fromarray(X_compressed[:, 0])
    # im.save("your_file.jpeg")

    # check/create the Output dir to write out the compressed image
    current_dir = os.getcwd()
    out_dir = current_dir+'Output'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # should write the compresses images, the data set to <out_dir>
    # how output all images? to Outout dir? 1 by 1?
    no_col = (X_compressed.shape)[1]
    for col in range(no_col):

        #compressed_img = X_compressed[:,col]
        #print(X_compressed)
        img_out = X_compressed[:, col].reshape((60, 48))
        # scaler = MinMaxScaler(feature_range=(0, 250))
        # scaler = scaler.fit(img_out)
        # X_scaled = scaler.transform(img_out)
        filename = str(col) + ".png"
        plt.imsave(os.path.join(out_dir, filename), img_out, cmap='gray', vmin=0, vmax=1)

    return 0



def load_data(input_dir):
    img_pattern = '*.pgm'
    images = os.path.join(input_dir, img_pattern)
    image_list = glob.glob(images)

    # sample_img_array = plt.imread(image_list[0])
    # sample_img_flat = sample_img_array.flatten()
    # no_of_img_pixels = len(sample_img_flat)

    DATA = []
    for img in image_list:
        img_array = plt.imread(img)
        # print(img_array.shape)
        img_flat = img_array.flatten()
        DATA.append(img_flat)
    DATA = np.array(DATA, dtype=float).transpose()

    return DATA



#
if __name__ == '__main__':

    pca_section = 'app' # 'app' OR 'pca'

    if (pca_section == 'app'):
        k = 1
        #input_dir = '/Users/ehsanmos/Documents/CS_courses_UNR/Fall2019/Machine_learning/Projects/Project_4/Data/Train'
        input_dir = './Data/Train/'
        X = load_data(input_dir)

        compress_images(X, 1000)




        # image_sample = '/Users/ehsanmos/Documents/CS_courses_UNR/Fall2019/Machine_learning/Projects/Project_4/Data/Test/00423_940422_fb.pgm'
        # im = cv2.imread(image_sample, -1)
        # type(im)
        # DATA = np.ndarray.flatten(im) # take the flattened image data (DATA) - image-flatten-array
        # compress_images(DATA, k)





    else:

        X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
        features = ['sepal length', 'sepal width', 'petal length', 'petal width']
        X = df.loc[:, features].values
        #X = np.array([[1,1],[1,0],[2,2],[2,1],[2,4],[3,4],[3,3],[3,2],[4,4],[4,5],[5,5],[5,7],[5,4]])

        # df = pd.read_csv(
        #     filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        #     header=None,
        #     sep=',')
        #
        # df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
        # df.dropna(how="all", inplace=True) # drops the empty line at file-end
        #
        # df.tail()
        # X = df.ix[:,0:4].values
        # y = df.ix[:,4].values
        #
        #

        Z = compute_Z(X,True,True)
        #Z = StandardScaler().fit_transform(X)
        COV = compute_covariance_matrix(Z)
        L, PCS = find_pcs(COV)

        print(L)
        print("#######")
        print(PCS)
        print("#######")
        Z_star = project_data(Z, PCS, L, 0, 0.95)
        #print(Z_star)
        print("#######")

        pca = PCA(n_components=4)
        principalComponents = pca.fit_transform(Z)
        print(pca.explained_variance_)
        print("#######")
        print(pca.components_)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = ['principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

        #print(finalDf)
