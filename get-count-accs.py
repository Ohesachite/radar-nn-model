import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from scipy.signal import correlate
from scipy.interpolate import RectBivariateSpline
from sklearn.cluster import DBSCAN
import csv
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as anim
import time
import json
import os

def load_point_cloud_vid(csvpaths, offsets, fps=16, eps=0.06, min_samples=5):
    if len(csvpaths) == 0:
        print("No point clouds to load")
        return
    assert len(csvpaths) == len(offsets)
    
    data_list = []
    for i, csvpath in enumerate(csvpaths):
        with open(csvpath) as csvfile:
            data = np.array(list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)))
            data[:,1] -= np.amin(data[:,1])
            
            offset_rot = offsets[i][3]
            offset_trans = offsets[i][0:3]
            
            data[:,3:6] = np.matmul(data[:,3:6], np.array([[np.cos(offset_rot), np.sin(offset_rot), 0], 
                                                [- np.sin(offset_rot), np.cos(offset_rot), 0], 
                                                [0, 0, 1]])) + np.array(offset_trans)
            
            data[:,7] /= 10000
            
            data_list.append(data)
            
    data_all = np.concatenate(data_list, axis=0)
    
    frame_period = 1.0 / fps
    max_time = np.amax(data_all[:,0])
    n_frames = int(max_time // frame_period + 1)
    frames_skipped = 0
    
    video = []
    for frame_n in range(n_frames):
        frame_mask = np.logical_and(frame_n * frame_period <= data_all[:,0], data_all[:,0] < (frame_n + 1) * frame_period)
        if not np.any(frame_mask):
            print("Zero point frame detected")
            frames_skipped += 1
            continue
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data_all[frame_mask, 3:6])
        if all(dbscan.labels_ == -1):
            print("DBSCAN reduced frame to zero points, dropping frame")
            frames_skipped += 1
            continue
        frame_points = data_all[frame_mask,2:][dbscan.labels_ != -1, :]
        frame_indicator = np.ones((frame_points.shape[0], 1)) * (frame_n - frames_skipped)
        frame = np.concatenate((frame_indicator, frame_points), axis=1)
        
        video.append(frame)
        
    return video

def median_centering(point_cloud_vid):
    all_points = np.concatenate(point_cloud_vid)
    median_point = np.median(all_points[:,2:5], axis=0)
    new_point_cloud_vid = []
    for n, pc in enumerate(point_cloud_vid):
        new_pc = np.copy(pc)
        new_point_cloud_vid.append(new_pc)
        new_point_cloud_vid[n][:,2:4] = new_pc[:,2:4] - median_point[:2]
    return new_point_cloud_vid, median_point

def find_boundaries(ssm_peak_matrix, decay=0.002):
    num_peaks = ssm_peak_matrix.shape[0]
    assert ssm_peak_matrix.shape[1] == num_peaks
    
    peaks_in_order = np.argsort(np.diag(ssm_peak_matrix))[::-1]
    
    candidate_regions = []
    
    for peak in peaks_in_order:
        
        if len(candidate_regions) == 0:
            candidate_regions.append([peak])
            continue
            
        if peak < candidate_regions[0][0]:
            if ssm_peak_matrix[peak, candidate_regions[0][0]] < 0.0:
                candidate_regions.insert(0, [peak])
            else:
                candidate_regions[0].insert(0, peak)
        elif peak > candidate_regions[-1][-1]:
            if ssm_peak_matrix[peak, candidate_regions[-1][-1]] < 0.0:
                candidate_regions.append([peak])
            else:
                candidate_regions[-1].append(peak)
        else:
            for ri in range(0, len(candidate_regions)):
                if ri != 0 and candidate_regions[ri-1][-1] < peak < candidate_regions[ri][0]:
                    corr_peak_l = ssm_peak_matrix[peak, candidate_regions[ri-1][-1]]
                    corr_peak_r = ssm_peak_matrix[peak, candidate_regions[ri][0]]
                    
                    if corr_peak_l < 0.0 and corr_peak_r < 0.0:
                        candidate_regions.insert(ri, [peak])
                    elif corr_peak_r > corr_peak_l:
                        candidate_regions[ri].insert(0, peak)
                    else:
                        candidate_regions[ri-1].append(peak)
                        
                    break
                elif candidate_regions[ri][0] < peak < candidate_regions[ri][-1]:
                    for wri in range(1, len(candidate_regions[ri])):
                        if candidate_regions[ri][wri-1] < peak < candidate_regions[ri][wri]:
                            corr_peak_l = ssm_peak_matrix[peak, candidate_regions[ri][wri-1]]
                            corr_peak_r = ssm_peak_matrix[peak, candidate_regions[ri][wri]]
                            
                            new_region_l = candidate_regions[ri][:wri]
                            new_region_r = candidate_regions[ri][wri:]
                            
                            if corr_peak_l < 0.0 and corr_peak_r < 0.0:
                                del candidate_regions[ri]
                                candidate_regions.insert(ri, new_region_r)
                                candidate_regions.insert(ri, [peak])
                                candidate_regions.insert(ri, new_region_l)
                            elif corr_peak_r > 0.0 and corr_peak_l < 0.0:
                                del candidate_regions[ri][:wri]
                                candidate_regions[ri].insert(0, peak)
                                candidate_regions.insert(ri, new_region_l)
                            elif corr_peak_r < 0.0 and corr_peak_l > 0.0:
                                del candidate_regions[ri][:wri]
                                candidate_regions.insert(ri, new_region_l)
                                candidate_regions[ri].append(peak)
                            else:
                                candidate_regions[ri].insert(wri, peak)
                                    
                            break
                    break
    
    num_regions = len(candidate_regions)
    region_score = np.zeros((num_regions, num_regions))
    
    for i in range(num_regions):
        region_r = ssm_peak_matrix[candidate_regions[i],:]
        for j in range(num_regions):
            region_rc = region_r[:,candidate_regions[j]]
            region_score[i,j] = np.sum(region_rc)
    
    def best_boundaries(region_scores, decay_factor=decay):
        num_regions = len(region_scores)
        seg_scores = np.zeros((num_regions // 2 + 1, num_regions))
        best_prev_up_index = -np.ones((num_regions // 2 + 1, num_regions, 2), dtype=int)
        
        memoized_block_scores = np.zeros(region_scores.shape)
        
        for pot_up in range(num_regions):
            for pot_down in range(num_regions):
                memoized_block_scores[pot_up, pot_down] = region_scores[pot_up, pot_up] + region_scores[pot_down, pot_down] - 2*region_scores[pot_up, pot_down]
        
        for pot_down in range(1, num_regions):
            best_up = np.argmax(memoized_block_scores[:pot_down, pot_down])
            best_prev_up_index[1, pot_down, 1] = best_up
            seg_scores[1, pot_down] = memoized_block_scores[best_up, pot_down]
            
        for segn in range(2, num_regions // 2 + 1):
            for pot_down in range(segn * 2 - 1, num_regions):
                up_vect = np.repeat(np.expand_dims(memoized_block_scores[(segn-1)*2:pot_down, pot_down], axis=0), pot_down-(segn-1)*2, axis=0)
                prev_vect = np.repeat(np.expand_dims(seg_scores[segn-1, (segn-1)*2-1:pot_down-1], axis=1), pot_down-(segn-1)*2, axis=1)
                sum_matrix = np.triu(up_vect + prev_vect)
                
                unadj_prev_up = np.unravel_index(np.argmax(sum_matrix), sum_matrix.shape)
                best_prev_up = (unadj_prev_up[0] + (segn-1)*2 - 1, unadj_prev_up[1] + (segn-1)*2)
                best_prev_up_index[segn, pot_down, :] = best_prev_up
                
                seg_scores[segn, pot_down] = seg_scores[segn-1, best_prev_up[0]] + memoized_block_scores[best_prev_up[1], pot_down]
                
        for segn in range(num_regions // 2 + 1):
            seg_scores[segn,:] *=  (1 - decay_factor)**segn
            
        best_seg_index = np.unravel_index(np.argmax(seg_scores), seg_scores.shape)
        best_seg = []
        
        curr_down = best_seg_index[1]
        for segn in range(best_seg_index[0], 0, -1):
            curr_prev_up = best_prev_up_index[segn, curr_down, :]
            best_seg.append([curr_prev_up[1], curr_down])
            curr_down = curr_prev_up[0]
        
        return best_seg
    
    rn_boundaries = best_boundaries(region_score)
    
    peak_boundaries = []
    
    for rnb in rn_boundaries[::-1]:
        peak_boundaries += [[candidate_regions[rnb[0]][0], candidate_regions[rnb[1]][-1]]]
            
    return peak_boundaries

def counting_loop(sets, decay=0.002):
    count_acc = {}

    file_names = [os.path.join("data/radar/", dataset, file) for dataset in sets for file in os.listdir("data/radar/" + dataset) if (file.endswith(".csv") and not file == os.path.basename("cycle_data.csv"))]
    for file_name in file_names:
        print(file_name)
        point_cloud_vid = load_point_cloud_vid([file_name], [[0.0, 0.0, 0.0, 0.0]])
        
        start_time = time.time()
        vid_len = len(point_cloud_vid)
        new_point_cloud_vid, median_point = median_centering(point_cloud_vid)
        fft_size = 64
        dim_min, dim_max = -1.2, 1.2 - 2.4/fft_size
        xi, yi, zi = np.meshgrid(np.linspace(dim_min, dim_max, fft_size), np.linspace(dim_min, dim_max, fft_size), np.linspace(dim_min, dim_max, fft_size))
        doppler_vals_over_time = []
        spatial_vals_over_time = []
        for pc in new_point_cloud_vid:
            doppler_vals = griddata(pc[:,2:5], pc[:,5], (xi, yi, zi), method='linear', fill_value=0.0)
            # spatial_vals = griddata(pc[:,2:5], np.ones(pc.shape[0]), (xi, yi, zi), method='linear', fill_value=0.0)
            doppler_vals_over_time.append(doppler_vals)

        doppler_vals_over_time = np.array(doppler_vals_over_time)
        end_time = time.time()
        print("Interpolation step done in", end_time - start_time, "seconds")

        window_size = 24
        template_found = False
        initial_template_size = 64
        template_alpha = 0.5

        segment_index = 0
        max_score = - np.inf
        max_score_props = (-1, -1)
        count = 0
        
        scores = []

        start_time = time.time()
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
                
        end_time = time.time()
        print("Correlation took", end_time-start_time, "seconds")
        
        fig_file_name = file_name.split('/')[2] + '_' + file_name.split('/')[3].split('.')[0]

        start_time = time.time()
        ssm_diag = np.diag(ssm_save)
        ssm_diag_peaks, _ = sp.signal.find_peaks(ssm_diag)
        if len(ssm_diag_peaks) > 0:
            ssm_peak_matrix = np.diag(ssm_diag[ssm_diag_peaks])
            for r in range(ssm_peak_matrix.shape[0]):
                for c in range(ssm_peak_matrix.shape[1]):
                    ssm_peak_matrix[r,c] = ssm_save[ssm_diag_peaks[r], ssm_diag_peaks[c]]

            bounds = find_boundaries(ssm_peak_matrix, decay=decay)

            end_time = time.time()
            print("Segmentation took", end_time-start_time, "seconds")

            n_excluded_bounds = 0
            for i in range(len(bounds)):
                bound_l, bound_r = bounds[i]
                if bound_l == 0:
                    sbl = (ssm_diag_peaks[bound_l]) // 2
                else:
                    sbl = (ssm_diag_peaks[bound_l] + ssm_diag_peaks[bound_l-1]) // 2 + 1

                if bound_r == len(ssm_diag_peaks) - 1:
                    sbr = (ssm_diag_peaks[bound_r] + vid_len) // 2
                else:
                    sbr = (ssm_diag_peaks[bound_r] + ssm_diag_peaks[bound_r+1]) // 2
                    
                if sbr - sbl < 8:
                    n_excluded_bounds += 1
                    continue
            
            count = len(bounds) - n_excluded_bounds
            exercise = file_name.split('/')[-1].split('.')[0].split('_')[1]
            
            print(exercise, count)
            set_num = file_name.split('/')[-2].split('_')[0][3:]
            
            if set_num == '8':
                if exercise == 'idle':
                    pgtc = [0]
                elif exercise == 'aud' or exercise == 'lud':
                    pgtc = [3, 6]
                else:
                    pgtc = [3]
            elif set_num == '9' or set_num == 'env-935' or set_num == 'env-5158':
                if exercise == 'idle':
                    pgtc = [0]
                elif exercise == 'aud' or exercise == 'lud':
                    pgtc = [5, 10]
                else:
                    pgtc = [5]
            elif set_num == '11':
                if exercise == 'idle':
                    pgtc = [0]
                elif exercise == 'aud' or exercise == 'lud':
                    pgtc = [10, 20]
                else:
                    pgtc = [10]
            else:
                assert False, "Using not set 8 or 9"
            
            if exercise not in count_acc:
                count_acc[exercise] = [0, 0, 0]
                
            gtc = pgtc[min(range(len(pgtc)), key = lambda i: abs(pgtc[i]-count))]
            
            count_acc[exercise][0] += 1
            if gtc != 0:
                count_acc[exercise][1] += abs(count - gtc) / gtc
                count_acc[exercise][2] += (abs(count - gtc) / gtc) ** 2
            
        else:
            end_time = time.time()
            print("No peaks were large enough, so terminating after", end_time-start_time, "seconds")
            
            count = 0
            exercise = file_name.split('/')[-1].split('.')[0].split('_')[1]
            print(exercise, count)
            set_num = file_name.split('/')[-2].split('_')[0][3:]
            
            if set_num == '8':
                if exercise == 'idle':
                    pgtc = [0]
                elif exercise == 'aud' or exercise == 'lud':
                    pgtc = [3, 6]
                else:
                    pgtc = [3]
            elif set_num == '9' or set_num == 'env-935' or set_num == 'env-5158':
                if exercise == 'idle':
                    pgtc = [0]
                elif exercise == 'aud' or exercise == 'lud':
                    pgtc = [5, 10]
                else:
                    pgtc = [5]
            elif set_num == '11':
                if exercise == 'idle':
                    pgtc = [0]
                elif exercise == 'aud' or exercise == 'lud':
                    pgtc = [10, 20]
                else:
                    pgtc = [10]
            else:
                assert False, "Using not set 8 or 9"
            
            if exercise not in count_acc:
                count_acc[exercise] = [0, 0, 0]
                
            gtc = pgtc[min(range(len(pgtc)), key = lambda i: abs(pgtc[i]-count))]
            
            count_acc[exercise][0] += 1
            if gtc != 0:
                count_acc[exercise][1] += abs(count - gtc) / gtc
                count_acc[exercise][2] += ((count - gtc) / gtc) ** 2
            
    return count_acc

def save_counts(count_accs, args):
    with open(args.br_results_path, 'w') as counts_file:
        json.dump(count_accs, counts_file)

    mae_accs = {}
    for key in count_accs:
        mae_accs[key] = [count_accs[key][1] / count_accs[key][0], (count_accs[key][2] / count_accs[key][0]) - (count_accs[key][1] / count_accs[key][0]) ** 2]

    with open(args.ov_results_path, 'w') as accs_file:
        json.dump(mae_accs, accs_file)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Counting model')

    parser.add_argument('--br-results-path', default='results/counts.json', type=str)
    parser.add_argument('--ov-results-path', default='results/count_accs.json', type=str)
    parser.add_argument('--sets', nargs='+', default=['set8_0', 'set8_60', 'set8_120', 'set8_180', 'set8_240', 'set8_300'], type=str)
    parser.add_argument('--decay', default=0.002, type=float)

    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":
    args = parse_args()
    count_accs = counting_loop(args.sets, decay=args.decay)
    print(count_accs)
    save_counts(count_accs, args)
