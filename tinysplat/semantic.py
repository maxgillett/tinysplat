import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

class SemanticSegmenter:
    def __init__(self, scene, load_model=False, skip_init=False, **kwargs):
        self.scene = scene
        self.device = kwargs['device']

        # Load semantic segmentation maps if they already exist
        stored_semantic = dict()
        dir_name = kwargs['semantic_path']
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        elif os.path.exists(dir_name):
            for file_name in tqdm(os.listdir(dir_name)):
                if file_name.endswith('.npy'):
                    stored_semantic[file_name[:-4]] = np.load(os.path.join(dir_name, file_name), allow_pickle=True)

        # Load the model if not all images have been processed
        if len(stored_semantic) < len(scene.cameras) or load_model:
            self.load_model(kwargs['semantic_model'])

        if skip_init: return

        for camera in tqdm(scene.cameras):
            semantic = stored_semantic.get(camera.name)
            if semantic is not None:
                camera.semantic_map = semantic
            else:
                semantic = self.estimate(camera)
                camera.semantic_map = semantic
                np.save(os.path.join(dir_name, camera.name + '.npy'), semantic)

    def load_model(self, depth_model="facebook/mask2former-swin-large-ade-semantic"):
        # load Mask2Former fine-tuned on ADE20k semantic segmentation
        self.processor = AutoImageProcessor.from_pretrained(depth_model)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(depth_model)

    def estimate(self, camera):
        image = camera.image.pil_image
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_semantic_map = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        return predicted_semantic_map


