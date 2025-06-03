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

        
        grayscale_image = (0.2989 * image[:, :, 0] +  # Red channel
                           0.5870 * image[:, :, 1] +  # Green channel
                           0.1140 * image[:, :, 2]).astype(image.dtype) # Blue channel

        # Set all color channels to the grayscale value
        image[:, :, 0] = grayscale_image  # Red channel
        image[:, :, 1] = grayscale_image  # Green channel
        image[:, :, 2] = grayscale_image  # Blue channel

        return Frame(image, data, 'RGB') 

    def shutdown(self): 
        print('MyFilter shutting down')

if __name__ == '__main__':
    Filter.run_multi([
        (VideoIn,  dict(sources='file://video.mp4!sync', outputs='tcp://*:5555')),  

        (MyFilter, dict(sources='tcp://localhost:5555',  outputs='tcp://*:5552', my_happy_little_option='YAY')),

        
        (Webvis,  dict(sources='tcp://localhost:5552')),
    ])