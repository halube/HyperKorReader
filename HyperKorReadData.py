#!/usr/bin/env python
# coding: utf-8

# In[165]:


import spectral as sp
import matplotlib.pyplot as plt
import numpy as np
import cv2
from plantcv import plantcv as pcv
import csv
import numpy as np
import os
import datetime
import re
import os
import glob


# Define the path to the desired directory
new_directory = "E:\\Hyperkor\\"

# Change the working directory to the new directory
os.chdir(new_directory)


class HyperKor_PAMHSI_Reader:
    PAM = None
    HSI = None
    RGB = None
    Test= None
    sp.settings.envi_support_nonlowercase_params = "True"

    def __init__(self,directory, suffix, PAM=True, RGB=True, HSI=True,Test=True,debug=False):
        self.debug = debug
        if self.debug:
            print("Reading & processing data of:", suffix)
        if PAM:
            self.PAM = self.__PAMData(directory,suffix,debug)

        if HSI:
            self.HSI = self.__HSIData(directory,suffix,debug)

        if RGB:
            self.RGB = self.__RGBData(directory,suffix,debug)

        if Test:
            self.Test = self.__Testmask(directory,suffix,debug)

    class __PAMData:
        def __init__(self,directory, suffix,debug):
            self.debug = debug
            if self.debug:
                print("Reading PAM-Data...", end='')
            prefixes = ['APH', 'CHL', 'PMT','ADP']
            shapes = [(2240, 2240, 2), (2240, 2240, 2), (2240, 2240, 9),(2240,2240,61)]
            data_list = []
            data_list2 = []
            for prefix, shape in zip(prefixes, shapes):
                file_path = f'./{directory}/CHLF_PAM_Images/{prefix}_{suffix}.DAT'
                if prefix != 'ADP':
                    try:
                        data = np.fromfile(file_path, dtype=np.uint16)
                        data_list.append(data.reshape(shape, order="F"))
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")
                        # Not using continue here, just print the error and handle it outside the loop if necessary
                else:
                    try:
                        data2 = np.fromfile(file_path, dtype=np.uint16)
                        data_list2.append(data2.reshape(shape, order="F"))
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")
                        # Skip missing 'ADP' file if data_list2 is empty
                        if not data_list2:
                            continue
            
            # Concatenate data if data_list is not empty
            if data_list:
                concatenated_data = np.concatenate(data_list, axis=2)
                self.PAM = concatenated_data
            else:
                self.PAM = None  # or handle it according to your needs
            
            # Process data_list2 if not empty and handle additional data
            if data_list2:
                concatenated_data2 = np.concatenate(data_list2, axis=2)
                self.raw_kineticTiminingFrame = concatenated_data2
            
                # Extract values from the HDR file
                values = []
                try:
                    with open(f'./{directory}/CHLF_PAM_Images/HDR_{suffix}.INF', 'r') as file:
                        for line in file:
                            key, value = line.split('=')
                            if "TmPamTimePoint" in key:
                                values.append(int(value.strip()))
                    self.raw_kineticTiminingFrame_names = values
                except FileNotFoundError:
                    print(f"HDR file not found: ./{directory}/CHLF_PAM_Images/HDR_{suffix}.INF")
            else:
                self.raw_kineticTiminingFrame = None  # or handle it according to your needs
            
            # Optionally handle what happens if concatenated_data2 is not set
            if not data_list2:
                self.raw_kineticTiminingFrame_names = None  # or handle it according to your needs
            self.raw_data = concatenated_data
            self.raw_data_names = ["Red", "FarRed", "Fbase", "Chlfl", "Fbase1", "F0", "Fm", "Fbase2", "Fs_prime", "Fm_prime", "F0_prime", "Fbase3", "Fbase4"]
            
            self.__calculate_parameters()

        def __calculate_parameters(self):
            if self.debug:
                print("Processing PAM-Data...", end='')
            blurred = cv2.GaussianBlur(self.raw_data[:, :, 3], (7, 7), 0)
            (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            Mask = threshInv/255
            Alpha = 1 - (cv2.divide(np.float32(self.raw_data[:, :, 0]), np.float32(self.raw_data[:, :, 1])))
            Alpha[Alpha<=0] = 0
            NDVI = cv2.divide(cv2.subtract(np.float32(self.raw_data[:, :, 1]), np.float32(self.raw_data[:, :, 0])),
                              cv2.add(np.float32(self.raw_data[:, :, 1]), np.float32(self.raw_data[:, :, 0])))
            NDVI[NDVI<=0] = 0
            Fv = cv2.subtract(np.float32(self.raw_data[:, :, 6]), np.float32(self.raw_data[:, :, 5]))
            #Fv[Fv<=0] = 0
            Fv_Fm = cv2.divide(Fv, np.float32(self.raw_data[:, :, 6]))
            Fv_Fm[Fv_Fm<=0] = 0
            Fv_Fm[Fv_Fm>=1] = np.nan
            Fv_prime=cv2.subtract(np.float32(self.raw_data[:, :, 9]), np.float32(self.raw_data[:, :, 10]))
            Fq_prime=cv2.subtract(np.float32(self.raw_data[:, :, 9]), np.float32(self.raw_data[:, :, 8]))
            Max_PSII=cv2.divide(np.float32(Fv_prime),np.float32(self.raw_data[:, :, 9]))
            Y_PSII=cv2.divide(np.float32(Fq_prime),np.float32(self.raw_data[:, :, 9]))
            Y_PSII[Y_PSII<=0] = 0
            Y_PSII[Y_PSII>=1] = np.nan
            NPQ=cv2.divide(cv2.subtract(np.float32(self.raw_data[:, :, 6]),np.float32(self.raw_data[:, :, 9])),np.float32(self.raw_data[:, :, 9]))
            NPQ[NPQ<=0] = 0
            NPQ[np.isinf(NPQ)] = np.nan
            NPQ[NPQ>=10] = np.nan
            qP=cv2.divide(np.float32(Fq_prime),np.float32(Fv_prime))
            qP[qP<=0] = 0
            qP[np.isinf(qP)] = np.nan
            qP[qP>=2] = np.nan
            qN=cv2.divide(cv2.subtract(np.float32(self.raw_data[:, :, 6]),np.float32(self.raw_data[:, :, 9])),cv2.subtract(np.float32(self.raw_data[:, :, 6]),np.float32(self.raw_data[:, :, 10])))
            qN[qN<=0] = 0
            qN[np.isinf(qN)] = np.nan
            qN[qN>=2] = np.nan
            qL=cv2.divide(np.float32(qP),cv2.divide(np.float32(self.raw_data[:, :, 10]),np.float32(self.raw_data[:, :, 8])))
            qL[qL<=0] = 0
            qL[np.isinf(qL)] = np.nan
            qL[qL>=2] = np.nan
            PAR=100
            FractionPSII=0.5
            ETR=cv2.multiply((Y_PSII*PAR*FractionPSII),Alpha)
            ETR[ETR<=0] = 0
            ETR[np.isinf(ETR)] = np.nan
            Helper_Y_NO_V1=cv2.divide(np.float32(self.raw_data[:, :, 6]), np.float32(self.raw_data[:, :, 5]))-1
            Helper_Y_NO_V2=1+cv2.add(NPQ,cv2.multiply(qL,Helper_Y_NO_V1))
            Y_NO=cv2.divide(1,Helper_Y_NO_V2)
            Y_NO[Y_NO<=0] = 0
            Y_NO[np.isinf(Y_NO)] = np.nan
            Y_NO[Y_NO>=1] = np.nan

            Y_NPQ=1-Y_PSII-Y_NO
            Y_NPQ[Y_NPQ<=0] = 0
            Y_NPQ[np.isinf(Y_NPQ)] = np.nan
            Y_NPQ[Y_NPQ>=1] = np.nan
            if self.debug:
                print("...done!")
            self.calculated_paramters= Mask, Alpha, NDVI, Fv, Fv_Fm,Fv_prime,Fq_prime,Y_PSII,NPQ,qP,qN,qL,ETR,Y_NO,Y_NPQ
            self.calculated_paramters=np.stack(self.calculated_paramters,axis=2)
            self.calculated_paramters_names= ["Mask","Alpha", "NDVI", "Fv", "Fv_Fm","Fv_prime","Fq_prime","Y_PSII","NPQ","qP","qN","qL","ETR","Y_NO","Y_NPQ","Max_PSII"]

   

    class __HSIData:
        def __init__(self,directory, suffix,debug):
            self.debug=debug
            if self.debug:
                print("Reading HSI-Data...", end='')
            file_path = f'./{directory}/HSI_RGB_Images/*{suffix}*.hdr'
            matching_files = glob.glob(file_path)
            #print(matching_files)
            try:
                HSI_raw = sp.envi.open(matching_files[0])
                wvl = HSI_raw.bands.centers
                rows, cols, bands = HSI_raw.nrows, HSI_raw.ncols, HSI_raw.nbands
                meta = HSI_raw.metadata
                if self.debug:
                    print("Correcting whiteblance:", end='')
                Ref = self.__find_closest_reference_file(directory,matching_files[0])
                if self.debug:
                    print("Identified wb_reference:", Ref, end='')
                HSI_raw_wb = sp.envi.open(f'./{directory}/HSI_RGB_Images/{Ref}')
                img = HSI_raw.load()
                img_ref = HSI_raw_wb.load()
                img_corrected = cv2.divide(np.float32(img), np.float32(img_ref))
                img_corrected = np.nan_to_num(img_corrected, posinf=0)
                img_corrected = np.rot90(img_corrected)
                HSI_corrected = np.clip(img_corrected, 0, 2)
                if self.debug:
                    print("...done!")
                self.wavelength, self.raw_data, self.wb_data, self.corrected_data, self.wb_name = wvl, HSI_raw, HSI_raw_wb, HSI_corrected, Ref
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                self.wavelength, self.raw_data, self.wb_data, self.corrected_data, self.ref_wb = None, None, None, None, None


        def __find_closest_reference_file(self,directory, file_path):
            target_file_path = file_path
            reference_directory = f"./{directory}/HSI_RGB_Images/"
            pattern = ".*_wb.hdr"
            target_modification_date = self.__get_modification_date(target_file_path)
    
            if target_modification_date is None:
                return None
            reference_files = [f for f in os.listdir(reference_directory) if
                               os.path.isfile(os.path.join(reference_directory, f)) and re.search(pattern, f)]
            earlier_reference_files = [f for f in reference_files if
                                       self.__get_modification_date(os.path.join(reference_directory, f)) < target_modification_date]
            if not earlier_reference_files:
                return None
            closest_reference_file = max(earlier_reference_files,
                                         key=lambda x: self.__get_modification_date(os.path.join(reference_directory, x)))
            return closest_reference_file

        def __get_modification_date(self, file_path):
            file_path = file_path
            try:
                # Get the modification time of the file in seconds since the epoch
                modification_time = os.path.getmtime(file_path)
                # Convert the modification time to a human-readable format
                modification_date = datetime.datetime.fromtimestamp(modification_time)
                return modification_date
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                return None

        
    class __RGBData:
        def __init__(self,directory, suffix,debug):
            self.debug=debug
            if self.debug:
                print("Reading RGB-Data...", end='')
            file_path = f'./{directory}/HSI_RGB_Images/*{suffix}*.png'
            matching_files = glob.glob(file_path)
            try:
                self.raw_data = cv2.imread(matching_files[0], cv2.IMREAD_COLOR)
                #print(matching_files[0])
                if self.debug:
                    print("...done!")
            except FileNotFoundError:
                if self.debug:
                    print(f"File not found: {matching_files[0]}")
                self.RGB = None
            try:
                if self.debug:
                    print("Correcting whiteblance for RGB:", end='')
                Ref2 = self.__find_closest_reference_file2(directory,matching_files[0])
                file_path = f'./{directory}/HSI_RGB_Images/{Ref2}'
                if self.debug:
                    print("Identified :",Ref2, end='')
                Ref2 = cv2.imread(file_path, cv2.IMREAD_COLOR).astype(np.float32)
                Raw_data32bit = self.raw_data.astype(np.float32)
        
                balanced = np.zeros_like(Raw_data32bit)
                for channel in range(3):  # Assuming RGB image
                    balanced[:,:,channel] = cv2.divide(Raw_data32bit[:,:,channel], Ref2[:,:,channel] + 1e-6)
        
                # Calculate mean values
                original_means = np.mean(Raw_data32bit, axis=(0, 1))
                balanced_means = np.mean(balanced, axis=(0, 1))
        
                # Adjust balanced image mean to match original image mean
                for channel in range(3):
                    balanced[:,:,channel] *= original_means[channel] / balanced_means[channel]
        
                self.balanced = np.clip(balanced, 0, 255).astype(np.uint8)
                if self.debug:
                    print("...done!")
            except FileNotFoundError:
                if self.debug:
                    print(f"File not found: {matching_files[0]}")
                self.RGB = None
            
        def __find_closest_reference_file2(self,directory, file_path):
                target_file_path = file_path
                reference_directory = f"./{directory}/HSI_RGB_Images/"
                pattern = ".*_wb.png"
                target_modification_date = self.__get_modification_date(target_file_path)
        
                if target_modification_date is None:
                    return None
                reference_files = [f for f in os.listdir(reference_directory) if
                        os.path.isfile(os.path.join(reference_directory, f)) and re.search(pattern, f)]
                earlier_reference_files = [f for f in reference_files if
                        self.__get_modification_date(os.path.join(reference_directory, f)) < target_modification_date]
                if earlier_reference_files == []:
                    later_files = [f for f in reference_files if
                                   self.__get_modification_date(os.path.join(reference_directory, f)) > target_modification_date]
                    # Sort files by modification date
                    later_files.sort(key=lambda f: self.__get_modification_date(os.path.join(reference_directory, f)))
                    return later_files[0]
                closest_reference_file = max(earlier_reference_files,
                        key=lambda x: self.__get_modification_date(os.path.join(reference_directory, x)))
                return closest_reference_file

        def __get_modification_date(self, file_path):
            file_path = file_path
            try:
                # Get the modification time of the file in seconds since the epoch
                modification_time = os.path.getmtime(file_path)
                # Convert the modification time to a human-readable format
                modification_date = datetime.datetime.fromtimestamp(modification_time)
                return modification_date
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                return None
    
    class __Testmask:
        def __init__(self, directory, suffix,debug):
            self.debug=debug
            if self.debug:
                print("Check if file is part of testset....", end='')
            file_path_PAM = f'./Evaluation/{directory}/Test/*PAM{suffix}*_mask.png'
            file_path_HSI = f'./Evaluation/{directory}/Test/*HSI{suffix}*_mask.png'
            file_path_RGB = f'./Evaluation/{directory}/Test/*RGB{suffix}*_mask.png'
            self.PAM_mask = cv2.imread(glob.glob(file_path_PAM)[0], cv2.IMREAD_GRAYSCALE) if glob.glob(file_path_PAM) else None
            self.HSI_mask = cv2.imread(glob.glob(file_path_HSI)[0], cv2.IMREAD_GRAYSCALE) if glob.glob(file_path_HSI) else None
            self.RGB_mask = cv2.imread(glob.glob(file_path_RGB)[0], cv2.IMREAD_GRAYSCALE) if glob.glob(file_path_RGB) else None
            if self.RGB_mask is not None:
                if self.debug:
                    print("yes", end='')
            else:
                if self.debug:
                    print("no", end='')
                delattr(self, "PAM_mask")
                delattr(self, "RGB_mask")
                delattr(self, "HSI_mask")
            if self.debug:
                print("...done!", end='')
            
                
def showColorimage(Bild):
    #Show the color-image with matplotlib
    plt.imshow(cv2.cvtColor(Bild, cv2.COLOR_BGR2RGB))
    plt.show()
def showGrayimage(Bild):
    #Show the gray-image with matplotlib
    plt.imshow(Bild, cmap=plt.cm.gray)
    plt.show()


# In[167]:


# ### Example usage
# suffix = 'E1535P0019N0001'
# directory="Experiment_V_A_thaliana_TestAllFrames"
# try:
#     HyperKor_PAMHSI = HyperKor_PAMHSI_Reader(directory=directory,suffix=suffix, PAM=True, RGB=True, HSI=True,debug=True)
# except ValueError:
#     pass  # Handle the exception if needed


# In[162]:





# In[71]:




