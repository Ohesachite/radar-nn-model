import os
import sys
import numpy as np
import csv
import json
from sklearn.cluster import DBSCAN
import scipy as sp
from scipy.interpolate import griddata

def process_radar_data(root, fps=16, train=True, eps=0.05, min_samples=3, radars=None, label_offset=None, count_metadata=False, testoutdir=None):

    point_clouds = {}
    count_data = {}

    with open(os.path.join(root, 'metadata.json'), 'r') as metadata:
        offset_map = json.load(metadata)

    for file_name in os.listdir(root):
        if file_name == "cycle_data.csv":
            with open(os.path.join(root, file_name)) as cycle_file:
                reader = csv.reader(cycle_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
                read_data = []
                cycle_data = {}
                for np1, np2, sta, end, cnt in reader:
                    values = [str(np1), str(np2), int(sta), int(end), int(cnt)]
                    read_data.append(values)
                if count_metadata:
                    for values in read_data:
                        cycle_data[(values[0], values[1])] = [values[2], values[3], values[4]]

        elif file_name.endswith(".csv"):
            name_parts = file_name.split('.')[0].split('_')
            if name_parts[0] == "label":
                with open(os.path.join(root, file_name)) as csv_file:
                    video = np.array(list(csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)))
                    video[:,7] /= 10000
                    video[:,6] /= 2

                if ((name_parts[1], name_parts[2][:2]) not in point_clouds.keys()) and ((ord(name_parts[2][2]) - ord('a')) in radars):
                    point_clouds[(name_parts[1], name_parts[2][:2])] = []

                for key in offset_map.keys():    
                    if key in name_parts[2] and (ord(key) - ord('a')) in radars:
                        if len(radars) > 1:
                            offsets = offset_map[key]
                            offset_rot = offsets[3]
                            offset_trans = offsets[:3]
                            video[:,3:6] = np.matmul(video[:,3:6], np.array([[np.cos(offset_rot), np.sin(offset_rot), 0], 
                                                    [- np.sin(offset_rot), np.cos(offset_rot), 0], 
                                                    [0, 0, 1]])) + np.array(offset_trans)
                            print("Detected key", key, "and giving offsets", offsets)
                        else:
                            print("Detected key", key, "but not offseting")
                        point_clouds[(name_parts[1], name_parts[2][:2])].append(video)
                        break

    comb_point_clouds = {}
    new_point_clouds = {}

    for key in point_clouds.keys():
        comb_point_clouds[key] = np.concatenate(point_clouds[key], axis=0)
        print("Initial number of points:", comb_point_clouds[key].shape[0])

        max_time = np.amax(comb_point_clouds[key][:,0])
        frame_period = 1 / fps
        n_frames = int(max_time // frame_period + 1)
        print("Number of frames:", n_frames)

        frames_skipped = 0

        new_point_clouds[key] = []
        for frame_n in range(n_frames):
            frame_mask = np.logical_and(frame_n * frame_period <= comb_point_clouds[key][:,0], comb_point_clouds[key][:,0] < (frame_n + 1) * frame_period)
            if comb_point_clouds[key][frame_mask, :].shape[0] == 0:
                print("Zero point frames detected from time", frame_n * frame_period, "to", (frame_n + 1) * frame_period)
                frames_skipped = frames_skipped + 1
                if count_metadata:
                    if key in cycle_data.keys():
                        if frame_n < cycle_data[key][0]:
                            cycle_data[key][0] = cycle_data[key][0] - 1
                        elif frame_n >= n_frames - cycle_data[key][1]:
                            cycle_data[key][1] = cycle_data[key][1] - 1
                continue
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(comb_point_clouds[key][frame_mask, 3:6])
            if sum(dbscan.labels_ != -1) <= 5:
                print("DBSCAN caused zero point frame from time", frame_n * frame_period, "to", (frame_n + 1) * frame_period, "! Dropping the frame!")
                frames_skipped = frames_skipped + 1
                if count_metadata:
                    if key in cycle_data.keys():
                        if frame_n < cycle_data[key][0]:
                            cycle_data[key][0] = cycle_data[key][0] - 1
                        elif frame_n >= n_frames - cycle_data[key][1]:
                            cycle_data[key][1] = cycle_data[key][1] - 1
                continue
            filtered_points = comb_point_clouds[key][frame_mask,2:][dbscan.labels_ != -1,:]
            frame_indicator = np.ones((filtered_points.shape[0], 1)) * (frame_n - frames_skipped)
            new_point_clouds[key].append(np.concatenate((frame_indicator, filtered_points), axis=1))
        
        new_point_clouds[key] = np.concatenate(new_point_clouds[key], axis=0)

        print("New number of points:", new_point_clouds[key].shape[0])
        if count_metadata:
            if key in cycle_data.keys():
                print("Start frame:", cycle_data[key][0], ", End frame:", n_frames - frames_skipped - cycle_data[key][1] - 1, ", Count:", cycle_data[key][2])

    if train:
        new_root = os.path.join(root, "../train")
    else:
        if testoutdir is not None:
            new_root = os.path.join(root, "..", testoutdir)
        else:
            new_root = os.path.join(root, "../test")

    if not os.path.exists(new_root):
        os.makedirs(new_root)

    if count_metadata:
        for key in new_point_clouds.keys():
            point_cloud_and_metadata = {}

            if label_offset is not None:
                if key[1][0] == '0':
                    label_num = int(key[1][1]) + label_offset
                else:
                    label_num = int(key[1]) + label_offset
                
                if label_num < 10:
                    new_label = '0' + str(label_num)
                else:
                    new_label = str(label_num)

                new_file_name = "label_" + key[0] + "_" + new_label + ".npz"
            else:
                new_file_name = "label_" + key[0] + "_" + key[1] + ".npz"

            with open(os.path.join(new_root, new_file_name), 'wb') as new_file:
                point_cloud_and_metadata["Point cloud"] = new_point_clouds[key]
                if key in cycle_data.keys():
                    point_cloud_and_metadata["Cycle data"] = cycle_data[key]
                else:
                    print("No cycle data for", key)
                np.save(new_file, point_cloud_and_metadata)

    else:
        # if os.path.isfile(os.path.join(new_root, "segment_boundaries.json")):
        #     with open(os.path.join(new_root, "segment_boundaries.json"), 'r') as old_segmentation_file:
        #         try:
        #             segments = json.load(old_segmentation_file)
        #         except json.decoder.JSONDecodeError:
        #             segments = {}
        # else:
        #     segments = {}
        if os.path.isfile(os.path.join(new_root, "segment_boundaries.json")):
            with open(os.path.join(new_root, "segment_boundaries.json"), 'r') as old_segmentation_file:
                try:
                    correlation_matrices = json.load(old_segmentation_file)
                except json.decoder.JSONDecodeError:
                    print("JSON Decode Error")
                    correlation_matrices = {}
        else:
            correlation_matrices = {}

        with open(os.path.join(new_root, "segment_boundaries.json"), 'w') as segmentation_file:

            for key in new_point_clouds.keys():
                # if not train:
                # bounds = run_segmentation_algorithm(new_point_clouds[key])
                correlation_matrix = run_segmentation_algorithm(new_point_clouds[key])

                if label_offset is not None:
                    if key[1][0] == '0':
                        label_num = int(key[1][1]) + label_offset
                    else:
                        label_num = int(key[1]) + label_offset
                    
                    if label_num < 10:
                        new_label = '0' + str(label_num)
                    else:
                        new_label = str(label_num)

                    new_file_name = "label_" + key[0] + "_" + new_label + ".npy"

                    # if not train:
                    # if key[0] not in segments.keys():
                        # segments[key[0]] = {}
                    # segments[key[0]][new_label] = bounds
                    if key[0] not in correlation_matrices.keys():
                        correlation_matrices[key[0]] = {}
                    correlation_matrices[key[0]][new_label] = correlation_matrix

                else:
                    new_file_name = "label_" + key[0] + "_" + key[1] + ".npy"

                    # if not train:
                    # if key[0] not in segments.keys():
                        # segments[key[0]] = {}
                    # segments[key[0]][key[1]] = bounds
                    if key[0] not in correlation_matrices.keys():
                        correlation_matrices[key[0]] = {}
                    correlation_matrices[key[0]][key[1]] = correlation_matrix

                with open(os.path.join(new_root, new_file_name), 'wb') as new_file:
                    np.save(new_file, new_point_clouds[key])

            # json.dump(segments, segmentation_file)
            json.dump(correlation_matrices, segmentation_file)
                

def run_segmentation_algorithm(cat_point_cloud_vid, grid_size=64):
    def median_centering(point_cloud_vid):
        all_points = np.concatenate(point_cloud_vid)
        median_point = np.median(all_points[:,2:5], axis=0)
        new_point_cloud_vid = []
        for n, pc in enumerate(point_cloud_vid):
            new_pc = np.copy(pc)
            new_point_cloud_vid.append(new_pc)
            new_point_cloud_vid[n][:,2:4] = new_pc[:,2:4] - median_point[:2]
            
        return new_point_cloud_vid, median_point

    # def find_boundaries(ssm_peak_matrix):
    #     num_peaks = ssm_peak_matrix.shape[0]
    #     assert ssm_peak_matrix.shape[1] == num_peaks
        
    #     peaks_in_order = np.argsort(np.diag(ssm_peak_matrix))[::-1]
        
    #     candidate_regions = []
        
    #     for peak in peaks_in_order:
            
    #         if len(candidate_regions) == 0:
    #             candidate_regions.append([peak])
    #             continue
                
    #         if peak < candidate_regions[0][0]:
    #             if ssm_peak_matrix[peak, candidate_regions[0][0]] < 0.0:
    #                 candidate_regions.insert(0, [peak])
    #             else:
    #                 candidate_regions[0].insert(0, peak)
    #         elif peak > candidate_regions[-1][-1]:
    #             if ssm_peak_matrix[peak, candidate_regions[-1][-1]] < 0.0:
    #                 candidate_regions.append([peak])
    #             else:
    #                 candidate_regions[-1].append(peak)
    #         else:
    #             for ri in range(0, len(candidate_regions)):
    #                 if ri != 0 and candidate_regions[ri-1][-1] < peak < candidate_regions[ri][0]:
    #                     corr_peak_l = ssm_peak_matrix[peak, candidate_regions[ri-1][-1]]
    #                     corr_peak_r = ssm_peak_matrix[peak, candidate_regions[ri][0]]
                        
    #                     if corr_peak_l < 0.0 and corr_peak_r < 0.0:
    #                         candidate_regions.insert(ri, [peak])
    #                     elif corr_peak_r > corr_peak_l:
    #                         candidate_regions[ri].insert(0, peak)
    #                     else:
    #                         candidate_regions[ri-1].append(peak)
                            
    #                     break
    #                 elif candidate_regions[ri][0] < peak < candidate_regions[ri][-1]:
    #                     for wri in range(1, len(candidate_regions[ri])):
    #                         if candidate_regions[ri][wri-1] < peak < candidate_regions[ri][wri]:
    #                             corr_peak_l = ssm_peak_matrix[peak, candidate_regions[ri][wri-1]]
    #                             corr_peak_r = ssm_peak_matrix[peak, candidate_regions[ri][wri]]
                                
    #                             new_region_l = candidate_regions[ri][:wri]
    #                             new_region_r = candidate_regions[ri][wri:]
                                
    #                             if corr_peak_l < 0.0 and corr_peak_r < 0.0:
    #                                 del candidate_regions[ri]
    #                                 candidate_regions.insert(ri, new_region_r)
    #                                 candidate_regions.insert(ri, [peak])
    #                                 candidate_regions.insert(ri, new_region_l)
    #                             elif corr_peak_r > 0.0 and corr_peak_l < 0.0:
    #                                 del candidate_regions[ri][:wri]
    #                                 candidate_regions[ri].insert(0, peak)
    #                                 candidate_regions.insert(ri, new_region_l)
    #                             elif corr_peak_r < 0.0 and corr_peak_l > 0.0:
    #                                 del candidate_regions[ri][:wri]
    #                                 candidate_regions.insert(ri, new_region_l)
    #                                 candidate_regions[ri].append(peak)
    #                             else:
    #                                 candidate_regions[ri].insert(wri, peak)
                                        
    #                             break
    #                     break
        
    #     num_regions = len(candidate_regions)
    #     region_score = np.zeros((num_regions, num_regions))
        
    #     for i in range(num_regions):
    #         region_r = ssm_peak_matrix[candidate_regions[i],:]
    #         for j in range(num_regions):
    #             region_rc = region_r[:,candidate_regions[j]]
    #             region_score[i,j] = np.sum(region_rc)
        
    #     def best_boundaries(region_scores, decay_factor=0.002):
    #         num_regions = len(region_scores)
    #         seg_scores = np.zeros((num_regions // 2 + 1, num_regions))
    #         best_prev_up_index = -np.ones((num_regions // 2 + 1, num_regions, 2), dtype=int)
            
    #         memoized_block_scores = np.zeros(region_scores.shape)
            
    #         for pot_up in range(num_regions):
    #             for pot_down in range(num_regions):
    #                 memoized_block_scores[pot_up, pot_down] = region_scores[pot_up, pot_up] + region_scores[pot_down, pot_down] - 2*region_scores[pot_up, pot_down]
            
    #         for pot_down in range(1, num_regions):
    #             best_up = np.argmax(memoized_block_scores[:pot_down, pot_down])
    #             best_prev_up_index[1, pot_down, 1] = best_up
    #             seg_scores[1, pot_down] = memoized_block_scores[best_up, pot_down]
                
    #         for segn in range(2, num_regions // 2 + 1):
    #             for pot_down in range(segn * 2 - 1, num_regions):
    #                 up_vect = np.repeat(np.expand_dims(memoized_block_scores[(segn-1)*2:pot_down, pot_down], axis=0), pot_down-(segn-1)*2, axis=0)
    #                 prev_vect = np.repeat(np.expand_dims(seg_scores[segn-1, (segn-1)*2-1:pot_down-1], axis=1), pot_down-(segn-1)*2, axis=1)
    #                 sum_matrix = np.triu(up_vect + prev_vect)
                    
    #                 unadj_prev_up = np.unravel_index(np.argmax(sum_matrix), sum_matrix.shape)
    #                 best_prev_up = (unadj_prev_up[0] + (segn-1)*2 - 1, unadj_prev_up[1] + (segn-1)*2)
    #                 best_prev_up_index[segn, pot_down, :] = best_prev_up
                    
    #                 seg_scores[segn, pot_down] = seg_scores[segn-1, best_prev_up[0]] + memoized_block_scores[best_prev_up[1], pot_down]
                    
    #         for segn in range(num_regions // 2 + 1):
    #             seg_scores[segn,:] *=  (1 - decay_factor)**segn
                
    #         best_seg_index = np.unravel_index(np.argmax(seg_scores), seg_scores.shape)
    #         best_seg = []
            
    #         curr_down = best_seg_index[1]
    #         for segn in range(best_seg_index[0], 0, -1):
    #             curr_prev_up = best_prev_up_index[segn, curr_down, :]
    #             best_seg.append([curr_prev_up[1], curr_down])
    #             curr_down = curr_prev_up[0]
            
    #         return best_seg
        
    #     rn_boundaries = best_boundaries(region_score)
        
    #     peak_boundaries = []
        
    #     if len(rn_boundaries) == 0:
    #         peak_boundaries += [[0, num_peaks-1]]
    #     else:
    #         for rnb in rn_boundaries[::-1]:
    #             peak_boundaries += [[candidate_regions[rnb[0]][0], candidate_regions[rnb[1]][-1]]]
            
    #     return peak_boundaries

    point_cloud_vid = []
    for frame_n in range(int(np.amax(cat_point_cloud_vid[:,0])+1)):
        point_cloud_vid.append(cat_point_cloud_vid[cat_point_cloud_vid[:,0] == frame_n,:])

    vid_len = len(point_cloud_vid)
    print(vid_len)
    new_point_cloud_vid, median_point = median_centering(point_cloud_vid)
    dim_min, dim_max = -1.2, 1.2 - 2.4/grid_size
    xi, yi, zi = np.meshgrid(np.linspace(dim_min, dim_max, grid_size), np.linspace(dim_min, dim_max, grid_size), np.linspace(dim_min, dim_max, grid_size))
    doppler_vals_over_time = []
    for pc in new_point_cloud_vid:
        doppler_vals = griddata(pc[:,2:5], pc[:,5], (xi, yi, zi), method='linear', fill_value=0.0)
        doppler_vals_over_time.append(doppler_vals)
    doppler_vals_over_time = np.array(doppler_vals_over_time)

    ssm_save = np.zeros((vid_len, vid_len))
    dv_txy = np.mean(doppler_vals_over_time, axis=3)
    dv_txz = np.mean(doppler_vals_over_time, axis=2)
    dv_tyz = np.mean(doppler_vals_over_time, axis=1)
    for i in range(vid_len):
        for j in range(vid_len):
            xy_correlation = np.mean(np.multiply(dv_txy[i,:,:], dv_txy[j,:,:]))
            xz_correlation = np.mean(np.multiply(dv_txz[i,:,:], dv_txz[j,:,:]))
            yz_correlation = np.mean(np.multiply(dv_tyz[i,:,:], dv_tyz[j,:,:]))
            
            ssm_save[i,j] = xy_correlation + xz_correlation + yz_correlation

    print("Correlation finished")

    return ssm_save.tolist()
            
    # ssm_diag = np.diag(ssm_save)
    # ssm_diag_peaks, _ = sp.signal.find_peaks(ssm_diag)
    # if len(ssm_diag_peaks) > 0:
    #     ssm_peak_matrix = np.diag(ssm_diag[ssm_diag_peaks])
    #     for r in range(ssm_peak_matrix.shape[0]):
    #         for c in range(ssm_peak_matrix.shape[1]):
    #             ssm_peak_matrix[r,c] = ssm_save[ssm_diag_peaks[r], ssm_diag_peaks[c]]

    #     bounds = find_boundaries(ssm_peak_matrix)
    #     print(bounds)
    #     print("Segmentation finished")

    #     final_bounds = []

    #     for i in range(len(bounds)):
    #         bound_l, bound_r = bounds[i]
    #         if bound_l == 0:
    #             sbl = (ssm_diag_peaks[bound_l]) // 2
    #         else:
    #             sbl = (ssm_diag_peaks[bound_l] + ssm_diag_peaks[bound_l-1]) // 2 + 1

    #         if bound_r == len(ssm_diag_peaks) - 1:
    #             sbr = (ssm_diag_peaks[bound_r] + vid_len) // 2
    #         else:
    #             sbr = (ssm_diag_peaks[bound_r] + ssm_diag_peaks[bound_r+1]) // 2

            
    #         if sbr - sbl < 8:
    #             continue

    #         region_in_bounds_order = np.argsort(np.diag(ssm_save)[sbl:sbr+1])[::-1]
    #         region_in_bounds_order += sbl
    #         region_in_bounds_order = region_in_bounds_order.tolist()

    #         final_bounds += [(int(sbl), int(sbr), region_in_bounds_order)]

    # else:
    #     print("No peaks were large enough, so terminating")

    #     region_in_bounds_order = np.argsort(np.diag(ssm_save))[::-1]
    #     region_in_bounds_order = region_in_bounds_order.tolist()

    #     final_bounds = [(int(0), int(vid_len-1), region_in_bounds_order)]

    # return final_bounds

if __name__ == "__main__":
    base_folder = '/home/alan/Documents/radar-nn-model/data/radar'
    # Set 8
    # process_radar_data(root=os.path.join(base_folder, 'set8_0'), eps=0.04, min_samples=3, radars=[0,1,2])                             # 1-3
    # process_radar_data(root=os.path.join(base_folder, 'set8_0'), eps=0.06, min_samples=5, radars=[0], label_offset=18)                # 19-21
    # process_radar_data(root=os.path.join(base_folder, 'set8_0'), eps=0.06, min_samples=5, radars=[1], label_offset=21)                # 22-24
    # process_radar_data(root=os.path.join(base_folder, 'set8_0'), eps=0.06, min_samples=5, radars=[2], label_offset=24)                # 25-27
    # process_radar_data(root=os.path.join(base_folder, 'set8_60'), eps=0.06, min_samples=5, train=False, radars=[0])                   # 1-3
    # process_radar_data(root=os.path.join(base_folder, 'set8_60'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=3)   # 4-6
    # process_radar_data(root=os.path.join(base_folder, 'set8_60'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=6)   # 7-9
    # process_radar_data(root=os.path.join(base_folder, 'set8_240'), eps=0.06, min_samples=5, train=False, radars=[0], label_offset=9)  # 10-12
    # process_radar_data(root=os.path.join(base_folder, 'set8_240'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=12) # 13-15
    # process_radar_data(root=os.path.join(base_folder, 'set8_240'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=15) # 16-18
    # process_radar_data(root=os.path.join(base_folder, 'set8_120'), eps=0.06, min_samples=5, train=False, radars=[0], label_offset=18) # 19-21
    # process_radar_data(root=os.path.join(base_folder, 'set8_120'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=21) # 22-24
    # process_radar_data(root=os.path.join(base_folder, 'set8_120'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=24) # 25-27
    # process_radar_data(root=os.path.join(base_folder, 'set8_300'), eps=0.06, min_samples=5, train=False, radars=[0], label_offset=27) # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'set8_300'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=30) # 31-33
    # process_radar_data(root=os.path.join(base_folder, 'set8_300'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=33) # 34-36
    # process_radar_data(root=os.path.join(base_folder, 'set8_180'), eps=0.06, min_samples=5, train=False, radars=[0], label_offset=36) # 37-39
    # process_radar_data(root=os.path.join(base_folder, 'set8_180'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=39) # 40-42
    # process_radar_data(root=os.path.join(base_folder, 'set8_180'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=42) # 43-45

    # Front testing
    # process_radar_data(root=os.path.join(base_folder, 'set9_0'), eps=0.06, min_samples=5, train=False, radars=[0])

    # Set 9
    # process_radar_data(root=os.path.join(base_folder, 'set9_0'), eps=0.04, min_samples=3, radars=[0,1,2], label_offset=3)             # 4-6
    # process_radar_data(root=os.path.join(base_folder, 'set9_0'), eps=0.06, min_samples=5, radars=[0], label_offset=27)                # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'set9_0'), eps=0.06, min_samples=5, radars=[1], label_offset=30)                # 31-33
    # process_radar_data(root=os.path.join(base_folder, 'set9_0'), eps=0.06, min_samples=5, radars=[2], label_offset=33)                # 34-36
    # process_radar_data(root=os.path.join(base_folder, 'set9_60'), eps=0.06, min_samples=5, train=False, radars=[0])                   # 1-3
    # process_radar_data(root=os.path.join(base_folder, 'set9_60'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=3)   # 4-6
    # process_radar_data(root=os.path.join(base_folder, 'set9_60'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=6)   # 7-9
    # process_radar_data(root=os.path.join(base_folder, 'set9_240'), eps=0.06, min_samples=5, train=False, radars=[0], label_offset=9)  # 10-12
    # process_radar_data(root=os.path.join(base_folder, 'set9_240'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=12) # 13-15
    # process_radar_data(root=os.path.join(base_folder, 'set9_240'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=15) # 16-18
    # process_radar_data(root=os.path.join(base_folder, 'set9_120'), eps=0.06, min_samples=5, train=False, radars=[0], label_offset=18) # 19-21
    # process_radar_data(root=os.path.join(base_folder, 'set9_120'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=21) # 22-24
    # process_radar_data(root=os.path.join(base_folder, 'set9_120'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=24) # 25-27
    # process_radar_data(root=os.path.join(base_folder, 'set9_300'), eps=0.06, min_samples=5, train=False, radars=[0], label_offset=27) # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'set9_300'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=30) # 31-33
    # process_radar_data(root=os.path.join(base_folder, 'set9_300'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=33) # 34-36
    # process_radar_data(root=os.path.join(base_folder, 'set9_180'), eps=0.06, min_samples=5, train=False, radars=[0], label_offset=36) # 37-39
    # process_radar_data(root=os.path.join(base_folder, 'set9_180'), eps=0.06, min_samples=5, train=False, radars=[1], label_offset=39) # 40-42
    # process_radar_data(root=os.path.join(base_folder, 'set9_180'), eps=0.06, min_samples=5, train=False, radars=[2], label_offset=42) # 43-45

    # C457 Set
    # process_radar_data(root=os.path.join(base_folder, 'setenv-457_0'), eps=0.04, min_samples=3, radars=[0,1,2], label_offset=36)               # 4-6
    # process_radar_data(root=os.path.join(base_folder, 'setenv-457_0'), eps=0.06, min_samples=5, radars=[0], label_offset=48)                   # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'setenv-457_0'), eps=0.06, min_samples=5, radars=[1], label_offset=60)                   # 31-33
    # process_radar_data(root=os.path.join(base_folder, 'setenv-457_0'), eps=0.06, min_samples=5, radars=[2], label_offset=72)                   # 34-36
    # process_radar_data(root=os.path.join(base_folder, 'setenv-457_seentar'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='test_2seen', label_offset=45)                   # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'setenv-457_seentar'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_2seen', label_offset=57)                   # 31-33
    # process_radar_data(root=os.path.join(base_folder, 'setenv-457_seentar'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_2seen', label_offset=69)                   # 34-36
    # process_radar_data(root=os.path.join(base_folder, 'setenv-457_unseentar'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='test_2seen', label_offset=45)                   # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'setenv-457_unseentar'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_2seen', label_offset=57)                   # 31-33
    # process_radar_data(root=os.path.join(base_folder, 'setenv-457_unseentar'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_2seen', label_offset=69)                   # 34-36

    # 5158 Set
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158-2_0'), eps=0.04, min_samples=3, radars=[0,1,2], label_offset=84)               # 4-6
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158-2_0'), eps=0.06, min_samples=5, radars=[0], label_offset=86)                   # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158-2_0'), eps=0.06, min_samples=5, radars=[1], label_offset=88)                   # 31-33
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158-2_0'), eps=0.06, min_samples=5, radars=[2], label_offset=90)                   # 34-36
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158-2_n0'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='test_seen3')                   # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158-2_n0'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_seen3', label_offset=2)                   # 31-33
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158-2_n0'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_seen3', label_offset=4)                   # 34-36

    # Environment Cross validation
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935s_0'), eps=0.05, min_samples=4, radars=[1,2], label_offset=36)
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935s_0'), eps=0.06, min_samples=5, radars=[1], label_offset=37)     # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935s_0'), eps=0.06, min_samples=5, radars=[2], label_offset=38)     # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158s_0'), eps=0.05, min_samples=4, radars=[1,2], label_offset=39)
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158s_0'), eps=0.06, min_samples=5, radars=[1], label_offset=40)    # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158s_0'), eps=0.06, min_samples=5, radars=[2], label_offset=41)    # 2

    # Unseen targets
    # process_radar_data(root=os.path.join(base_folder, 'setenv-131_far'), eps=0.08, min_samples=4, train=False, radars=[0], testoutdir='test_ut')    # 2
    # process_radar_data(root=os.path.join(base_folder, 'set-unseentar'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='test_ut')                   # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'set-unseentar'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_ut', label_offset=12)                   # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'set-unseentar'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_ut', label_offset=24)                   # 28-30

    # Unseen (all separated)
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935s_0'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_s1_0')                       # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935s_0'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_s1_0', label_offset=2)       # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935u_0'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_u1_0')                      # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935u_0'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_u1_0', label_offset=2)      # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935s_60'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_s1_60')                      # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935s_60'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_s1_60', label_offset=2)      # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935u_60'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_u1_60')                     # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935u_60'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_u1_60', label_offset=2)     # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158s_0'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_s2_0')                      # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158s_0'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_s2_0', label_offset=2)      # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158u_0'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_u2_0')                    # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158u_0'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_u2_0', label_offset=2)    # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158s_270'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_s2_270')                    # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158s_270'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_s2_270', label_offset=2)    # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158u_270'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_u2_270')                  # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158u_270'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_u2_270', label_offset=2)  # 2

    # Unseen (by environment)
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935s_0'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_ue_0', label_offset=0)     # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935s_0'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_ue_n0', label_offset=1)     # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935u_0'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_ue_0', label_offset=1)     # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935u_0'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_ue_n0', label_offset=2)     # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935s_60'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_ue_n0', label_offset=4)    # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935s_60'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_ue_n0', label_offset=5)    # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935u_60'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_ue_n0', label_offset=5)    # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-935u_60'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_ue_n0', label_offset=6)    # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158s_0'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_ue2_0', label_offset=0)    # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158s_0'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_ue2_n0', label_offset=1)    # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158u_0'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_ue2_0', label_offset=1)    # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158u_0'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_ue2_n0', label_offset=3)    # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158s_270'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_ue2_n0', label_offset=6)  # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158s_270'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_ue2_0', label_offset=7)  # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158u_270'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_ue2_n0', label_offset=7)  # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-5158u_270'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='test_ue2_0', label_offset=9)  # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-131_0'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='test_ue_0', label_offset=8)    # 1
    # process_radar_data(root=os.path.join(base_folder, 'setenv-131_n0'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='test_ue_n0', label_offset=8)    # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-131_far'), eps=0.08, min_samples=4, train=False, radars=[0], testoutdir='test_ue_n0', label_offset=8)    # 2
    # process_radar_data(root=os.path.join(base_folder, 'setenv-TSRB-conf'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='test_ue_n0', label_offset=12)

    # Strict orientation splits
    # process_radar_data(root=os.path.join(base_folder, 'set8_60'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='rtest_60')                    # 1-3
    # process_radar_data(root=os.path.join(base_folder, 'set8_60'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='rtest_300', label_offset=3)   # 4-6
    # process_radar_data(root=os.path.join(base_folder, 'set8_60'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='rtest_180', label_offset=6)   # 7-9
    # process_radar_data(root=os.path.join(base_folder, 'set8_240'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='rtest_240', label_offset=9)  # 10-12
    # process_radar_data(root=os.path.join(base_folder, 'set8_240'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='rtest_120', label_offset=12) # 13-15
    # process_radar_data(root=os.path.join(base_folder, 'set8_240'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='rtest_0', label_offset=15) # 16-18
    # process_radar_data(root=os.path.join(base_folder, 'set8_120'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='rtest_120', label_offset=18) # 19-21
    # process_radar_data(root=os.path.join(base_folder, 'set8_120'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='rtest_0', label_offset=21) # 22-24
    # process_radar_data(root=os.path.join(base_folder, 'set8_120'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='rtest_240', label_offset=24) # 25-27
    # process_radar_data(root=os.path.join(base_folder, 'set8_300'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='rtest_300', label_offset=27) # 28-30
    # process_radar_data(root=os.path.join(base_folder, 'set8_300'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='rtest_180', label_offset=30) # 31-33
    # process_radar_data(root=os.path.join(base_folder, 'set8_300'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='rtest_60', label_offset=33) # 34-36
    # process_radar_data(root=os.path.join(base_folder, 'set8_180'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='rtest_180', label_offset=36) # 37-39
    # process_radar_data(root=os.path.join(base_folder, 'set8_180'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='rtest_60', label_offset=39) # 40-42
    # process_radar_data(root=os.path.join(base_folder, 'set8_180'), eps=0.06, min_samples=5, train=False, radars=[2], testoutdir='rtest_300', label_offset=42) # 43-45

    # NLoS
    # process_radar_data(root=os.path.join(base_folder, 'set_10'), eps=0.06, min_samples=5, train=False, radars=[1], testoutdir='test_NLOS')
    # process_radar_data(root=os.path.join(base_folder, 'set_10'), eps=0.06, min_samples=5, train=False, radars=[0], testoutdir='test_nNLOS')