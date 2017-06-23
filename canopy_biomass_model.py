import numpy as np

import gdal
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from sklearn.ensemble import RandomForestRegressor


def open_array(chm_file):
    chm_dataset = gdal.Open(chm_file)
    chm_raster = chm_dataset.GetRasterBand(1)
    chm_array = chm_raster.ReadAsArray(0, 0, chm_dataset.RasterXSize,
                                      chm_dataset.RasterYSize).astype(np.float)
    return chm_array

def smooth_chm(chm_array):
    chm_array_smooth = ndi.gaussian_filter(chm_array, 2, mode='constant', cval=0, truncate=2.0)
    chm_array_smooth[chm_array == 0] = 0
    return chm_array_smooth

def get_trees(chm_array):
    chm_array_smooth = smooth_chm(chm_array)
    local_maxi = peak_local_max(chm_array_smooth, indices=False, footprint=np.ones((5, 5)))
    markers = ndi.label(local_maxi)[0]
    chm_mask = chm_array_smooth
    chm_mask[chm_array_smooth != 0] = 1
    labels = watershed(chm_array_smooth, markers, mask=chm_mask)
    tree_properties = regionprops(labels, chm_array,
                                ['Area', 'BoundingBox', 'Centroid',
                                 'Orientation', 'MajorAxisLength',
                                 'MinorAxisLength', 'MaxIntensity', 'MinIntensity'])
    return tree_properties, labels


def get_chm_predictors(chm_array):
    tree_properties, labels = get_trees(chm_array)
    return [get_predictors(tree, chm_array, labels) for tree in tree_properties], labels


def get_predictors(tree, chm_array, labels):
    indexes_of_tree = np.asarray(np.where(labels == tree.label)).T
    tree_data = chm_array[indexes_of_tree[:, 0], indexes_of_tree[:, 1]]
    full_crown = np.sum(tree_data - np.min(tree_data))

    def crown_geometric_volume_pth(pth):
        p = np.percentile(tree_data, pth)
        tree_data_pth = [v if v < p else p for v in tree_data]
        crown_geometric_volume_pth = np.sum(tree_data_pth - tree.min_intensity)
        return crown_geometric_volume_pth, p

    crown50, p50 = crown_geometric_volume_pth(50)
    crown60, p60 = crown_geometric_volume_pth(60)
    crown70, p70 = crown_geometric_volume_pth(70)

    return [tree.label,
            np.float(tree.area),
            tree.major_axis_length,
            tree.max_intensity,
            tree.min_intensity,
            p50, p60, p70,
            full_crown, crown50, crown60, crown70]


def make_canopy_biomass_model(training_data):
    biomass = training_data[:, 0]
    biomass_predictors = training_data[:, 1:12]
    max_depth = 30
    regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
    regr_rf.fit(biomass_predictors, biomass)
    return regr_rf

def apply_canopy_biomass_model(predictors, labels, regr_rf):
    chm_dependant_data = np.array([x[1:] for x in predictors])
    pred_biomass = regr_rf.predict(chm_dependant_data)
    print np.sum(pred_biomass)
    biomass_out = labels
    for p, bm in zip(predictors, pred_biomass):
        biomass_out[biomass_out == p[0]] = bm
    return biomass_out


chm_file = 'NEON_D17_SJER_DP3_256000_4106000_CHM.tif'
chm_array = open_array(chm_file)

training_data_file = 'SJER_Biomass_Training.csv'
training_data = np.genfromtxt(training_data_file, delimiter=',')

predictors, labels = get_chm_predictors(chm_array)


regr_rf = make_canopy_biomass_model(training_data)

pred_biomass = apply_canopy_biomass_model(predictors, labels, regr_rf)


mean_biomass = np.mean(pred_biomass)
std_biomass = np.std(pred_biomass)
min_biomass = np.min(pred_biomass)
sum_biomass = np.sum(pred_biomass)

print('Sum of biomass is ', sum_biomass, ' kg')

#('Sum of biomass is ', 6977394.0499962922, ' kg')

