import cv2
import os
import pickle

folder_dir = 'dataset/action_youtube_naudio'
class_list = sorted(os.listdir(folder_dir))
# print(class_list)
class_list.remove('readme.txt')
# print(class_list)

train_vids = {}
test_vids = {}

data_count = 0
train_count = 0
test_count = 0
not_count = 0

for c in class_list:
    class_path = os.path.join(folder_dir, c)
    video_c = os.listdir(class_path)
    video_c.remove('Annotation')
    video_c = sorted(video_c)
    for vid in video_c:
        # print(vid)
        vid_num = int(vid.split('_')[-1])
        # print(vid_num)
        video_path = os.path.join(class_path, vid)
        clip_list = sorted(os.listdir(video_path))
        for clip in clip_list:
            clip_path = os.path.join(video_path, clip)
            ext = clip.split('.')[-1]
            clip_name = clip.split('.')[:-1][0]
            # clip_path = os.path.join(video_path, clip_name)
            # print(clip_name)
            if ext == 'avi':
                data_count += 1
                read_clip = cv2.VideoCapture(clip_path)
                total_frames = read_clip.get(7)
                fps = read_clip.get(5)
                # if fps < 10:
                    # print(clip_path)
                    # print(fps)
                after_frames = int(total_frames/(fps/5))
                if after_frames >= 16:
                    clip_name = os.path.join(video_path, clip_name)
                    if vid_num <= 20:
                        train_count += 1
                        train_vids[clip_name] = class_list.index(c)
                    else:
                        test_count += 1
                        test_vids[clip_name] = class_list.index(c)
                else:
                    not_count += 1
                    # print(clip_path)
                    # print('not enough')
            else:
                print(clip_path)
                os.remove(clip_path)
    # print(train_vids)
    # print(test_vids)
print(data_count)
print(train_count, test_count)
print(not_count)
save_split = 'data_split.pkl'
save_data = {}
save_data['train'] = train_vids
save_data['test'] = test_vids
with open(save_split, 'wb') as sd:
    pickle.dump(save_data, sd)
