import os
import copy
import numpy as np
import cv2


class JigsawPuzzle():
    def __init__(self,complete_images_paths:list[str],size = (1500,1000), n_rows = 10, n_cols = 10):
        self.displayed_image = None
        self.size = size
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.row_n_pixels = self.size[0]//self.n_rows
        self.col_n_pixels = self.size[1]//self.n_rows


        self.stored_complete_images = {}
        self.stored_jigsaw_images = {} # name = i, j piece of image
        for img_path in complete_images_paths:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)
            name = os.path.basename(img_path)
            print("name",name,"img",img.shape)

            self.stored_complete_images[name] = img

            loc_to_piece_img = self.transform_jigsaw(img)
            self.stored_jigsaw_images[name] = loc_to_piece_img
        
        

        self.randomize_puzzle()
        self.plot()



    def transform_jigsaw(self,img)->dict:
        """
        
        Returns:
        --------
        - loc_to_piece_img (dict): 
            (i,j) to piece of image
        
        """

        if img.shape[:2] != self.size:
            img = cv2.resize(img, self.size, interpolation = cv2.INTER_LINEAR)
        
        

        loc_to_piece_img = {}
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                loc_to_piece_img[i,j] = img[self.col_n_pixels*j:self.col_n_pixels*(j+1),self.row_n_pixels*i:self.row_n_pixels*(i+1)]
        
        return loc_to_piece_img


    def randomize_puzzle(self):
        init_id_img = np.random.choice(np.arange(len(self.stored_complete_images)))
        self.displayed_image = copy.deepcopy(list(self.stored_complete_images.values())[init_id_img])
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                chosen_id_img = np.random.choice(np.arange(len(self.stored_complete_images)))
                name_img_chosen = list(self.stored_jigsaw_images.keys())[chosen_id_img]
                piece_chosen = self.stored_jigsaw_images[name_img_chosen][i,j]

                self.displayed_image[self.col_n_pixels*j:self.col_n_pixels*(j+1),self.row_n_pixels*i:self.row_n_pixels*(i+1)] = piece_chosen
                


    def flip_piece(self,rowi,colj):
        pass

    
    def save_img(self,dst_path):
        cv2.imwrite(dst_path, self.displayed_image)
    
    def save_transformed_images(self,dst_folder_path):
        for name in self.stored_complete_images:
            cv2.imwrite(os.path.join(dst_folder_path,name), self.stored_complete_images[name])


    def plot(self):
        cv2.imshow("image", self.displayed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    puzzle = JigsawPuzzle(complete_images_paths=[f"data/raw_stimuli/vg{i}.jpg" for i in range(1,5)],
                          n_rows=20,n_cols=20)
    puzzle.save_img("example_jigsaw.jpg")
    puzzle.save_transformed_images("data/examples/original_images")