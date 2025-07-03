from openfilter.filter_runtime import Frame, Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis

class QuadrantColorGrayFilter(Filter):
    def setup(self, config):
        print(f'QuadrantColorGrayFilter setup: {config.Quadrant_Filter=}')

    def process(self, frames):
        frame = frames['main'].rw_rgb
        image = frame.image
        data  = frame.data
        
        height, width, _ = image.shape # Get image dimensions

        # Calculate midpoints for dividing the image
        mid_h = height // 2
        mid_w = width // 2

        quadrant2 = image[0:mid_h, mid_w:width, :] # Slice the top-right quadrant

        
        grayscale_q2 = (0.2989 * quadrant2[:, :, 0] +  # Red channel of Q2
                        0.5870 * quadrant2[:, :, 1] +  # Green channel of Q2
                        0.1140 * quadrant2[:, :, 2]).astype(image.dtype) # Convert Quadrant 2 to grayscale

        # Assign the grayscale values back to all three channels of Quadrant 2
        image[0:mid_h, mid_w:width, 0] = grayscale_q2 # Red channel of Q2
        image[0:mid_h, mid_w:width, 1] = grayscale_q2 # Green channel of Q2
        image[0:mid_h, mid_w:width, 2] = grayscale_q2 # Blue channel of Q2

        quadrant3 = image[mid_h:height, 0:mid_w, :] # Slice the bottom-left quadrant

        grayscale_q3 = (0.2989 * quadrant3[:, :, 0] +  # Red channel of Q3
                        0.5870 * quadrant3[:, :, 1] +  # Green channel of Q3
                        0.1140 * quadrant3[:, :, 2]).astype(image.dtype) # Convert Quadrant 3 to grayscale

        # Assign the grayscale values back to all three channels of Quadrant 3
        image[mid_h:height, 0:mid_w, 0] = grayscale_q3 # Red channel of Q3
        image[mid_h:height, 0:mid_w, 1] = grayscale_q3 # Green channel of Q3
        image[mid_h:height, 0:mid_w, 2] = grayscale_q3 # Blue channel of Q3

        return Frame(image, data, 'RGB')

    def shutdown(self):
        print('QuadrantColorGrayFilter shutting down')

if __name__ == '__main__':
    Filter.run_multi([
        (VideoIn,  dict(sources='file://video.mp4!sync', outputs='tcp://*:5555')),  

        (QuadrantColorGrayFilter, dict(sources='tcp://localhost:5555',  outputs='tcp://*:5552', Quadrant_Filter='Done')),
        
        (Webvis,  dict(sources='tcp://localhost:5552')),
    ])