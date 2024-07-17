import os
import os.path as osp
import sys
import cv2
import time
from typing import List


class VideoFrameDataset:
    def __init__(self, all_sequence_dir, log=print, debug=False, frame_rate=30, play_dur=30):
        self.debug = debug
        self.log = log
        self.path = all_sequence_dir
        self.play_dur = play_dur
        self.frame_dur = 1./frame_rate
        self.sequence_dirs: List[str] = []
        for d in os.listdir(all_sequence_dir):
            frames_dir = osp.join(all_sequence_dir, d)
            self.sequence_dirs.append(frames_dir)
        log(f"Dataset all sequence_dir {all_sequence_dir} number of sequences {len(self.sequence_dirs)}")
        self.sequence_iter = iter(self.sequence_dirs)
        self.current_sequence: str = None
        self.current_sequence_frames: List[str] = []
        self.last_frame_idx = -1
        self.last_time = 0
        self.sequence_start_time = 0

    def __len__(self):
        return int(1e8)

    def next_sequence(self):
        # infinite loop
        try:
            self.current_sequence = next(self.sequence_iter)
        except StopIteration:
            self.sequence_iter = iter(self.sequence_dirs)
            self.current_sequence = next(self.sequence_iter)
        frames = []
        self.log(f"Current sequence {self.current_sequence}")
        for f in sorted(os.listdir(self.current_sequence)):
            if f.endswith((".jpg", ".png")):
                frames.append(osp.join(self.current_sequence, f))
        self.current_sequence_frames = frames

        self.sequence_start_time = time.time()


    def __getitem__(self, i):
        ctime = time.time()
        if not self.current_sequence or ctime - self.sequence_start_time > self.play_dur:
            self.next_sequence()
            ctime = self.sequence_start_time
            self.last_frame_idx = -1
        frame_idx = max(int((ctime - self.sequence_start_time)//self.frame_dur), self.last_frame_idx + 1)
        self.last_frame_idx = frame_idx
        sequence_len = len(self.current_sequence_frames)
        forward_backward = (frame_idx // sequence_len) % 2
        if forward_backward == 0:   # forward
            true_idx = frame_idx % sequence_len
        else:                       # backward
            true_idx = - (frame_idx % sequence_len)
        if self.debug:
            self.log(f"Reading image {self.current_sequence_frames[true_idx]}")
        return cv2.imread(self.current_sequence_frames[true_idx])[..., [2,1,0]] # BGR to RGB


def test(path):
    import matplotlib.pyplot as plt
    d = VideoFrameDataset(path)
    for i in range(20):
        img = d[0]
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    test("/home/guanxiux/project/ParallelCollaborativeInference/data/DAVIS/JPEGImages/1080p")
    test("/home/guanxiux/project/ParallelCollaborativeInference/data/ActivityData")