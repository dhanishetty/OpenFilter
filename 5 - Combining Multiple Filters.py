from openfilter.filter_runtime import Frame, Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis

class RGBFilter(Filter):
    def setup(self, config):
        print(f'RGBFilter setup: {config.First_Filter=}')

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
        print('RGBFilter shutting down')

class GrayFilter(Filter):
    def setup(self, config):  
        print(f'GrayFilter setup: {config.Second_Filter=}')

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
        print('GrayFilter shutting down')

class FlipFilter(Filter):
    def setup(self, config):
        print(f'MyFilter setup: {config.Third_Filter=}')

    def process(self, frames):
        frame = frames['main'].rw_rgb
        image = frame.image
        data  = frame.data
               
        flipped_image = image[:, ::-1, :] # Perform the horizontal flip
        
        
        return Frame(flipped_image, data, 'RGB')# Return the frame with the flipped image

    def shutdown(self):
        print('FlipFilter shutting down')

if __name__ == '__main__':
    Filter.run_multi([
        (VideoIn,  dict(sources='file://video.mp4!sync', outputs='tcp://*:5555')),  

        (RGBFilter, dict(sources='tcp://localhost:5555', outputs='tcp://*:5557', First_Filter='Done')),

        (GrayFilter, dict(sources='tcp://localhost:5557', outputs='tcp://*:5552', Second_Filter='Done')),

        (FlipFilter, dict(sources='tcp://localhost:5552', outputs='tcp://*:5559', Third_Filter='Done')),
        
        (Webvis,  dict(sources='tcp://localhost:5559')),
    ])