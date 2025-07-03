from openfilter.filter_runtime import Frame, Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis

class MyFilter(Filter):
    def setup(self, config):
        print(f'MyFilter setup: {config.my_happy_little_option=}')

    def process(self, frames):
        frame = frames['main'].rw_rgb
        image = frame.image
        data  = frame.data
               
        flipped_image = image[:, ::-1, :] # Perform the horizontal flip
        
        
        return Frame(flipped_image, data, 'RGB')# Return the frame with the flipped image

    def shutdown(self):
        print('MyFilter shutting down')

if __name__ == '__main__':
    Filter.run_multi([
        (VideoIn,  dict(sources='file://video.mp4!sync', outputs='tcp://*:5555')),  

        (MyFilter, dict(sources='tcp://localhost:5555', outputs='tcp://*:5552', my_happy_little_option='YAY')),
        
        (Webvis,  dict(sources='tcp://localhost:5552')),
    ])