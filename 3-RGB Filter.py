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

        # Get image dimensions
        height, width, channels = image.shape
        
        
        third_width = width // 3 # Calculate the width of each third

        
        image[:, :third_width, 0] = 0 # First third: Zero out Red channel
        
        image[:, third_width:2 * third_width, 1] = 0 # Middle third: Zero out Green channel       
       
        image[:, 2 * third_width:, 2] = 0  # Last third: Zero out Blue channel
        
        return Frame(image, data, 'RGB')

    def shutdown(self):
        print('MyFilter shutting down')

if __name__ == '__main__':
    Filter.run_multi([
        (VideoIn,  dict(sources='file://video.mp4!sync', outputs='tcp://*:5555')),  

        (MyFilter, dict(sources='tcp://localhost:5555', outputs='tcp://*:5552', my_happy_little_option='YAY')),
        
        (Webvis,  dict(sources='tcp://localhost:5552')),
    ])