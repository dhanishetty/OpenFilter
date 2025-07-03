import numpy as np
from PIL import Image #PIL = Python Imaging Library
import torchvision.transforms as transforms
from openfilter.filter_runtime import Frame, Filter
from openfilter.filter_runtime.filters.webvis import Webvis
from openfilter.filter_runtime.filters.video_in import VideoIn

class MyTorchTransformFilter(Filter):
    def setup(self, config):
        print(f'MyTorchTransformFilter setup: {config.transform_name=}')
        self.transform = transforms.Compose([                
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                transforms.ToTensor()# Convert to PyTorch Tensor (C, H, W, float 0-1)
            ])
        
    def process(self, frames):
        frame = frames['main'].rw_rgb
        image_np = frame.image 
        data = frame.data

        if image_np.dtype != np.uint8:
             image_np = image_np.astype(np.uint8)

        pil_image = Image.fromarray(image_np)# Convert NumPy array to PIL Image 

        transformed_tensor = self.transform(pil_image)# Apply the torchvision transform pipeline

        # Convert PyTorch Tensor back to NumPy array
        numpy_output_image = transformed_tensor.permute(1, 2, 0).numpy() # Change from C,H,W to H,W,C
        numpy_output_image = (numpy_output_image * 255).astype(np.uint8) # Scale to 0-255 and convert to uint8

        return Frame(numpy_output_image, data, 'RGB')

    def shutdown(self):
        print('MyTorchTransformFilter shutting down')

    
if __name__ == '__main__':
    Filter.run_multi([
        (VideoIn, dict(sources='file://video.mp4!sync', outputs='tcp://*:5555')),

        (MyTorchTransformFilter, dict(sources='tcp://localhost:5555', outputs='tcp://*:5552', transform_name='color_jitter')),

        (Webvis, dict(sources='tcp://localhost:5552')),
    ])