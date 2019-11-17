from PCV.tools import imtools, pca
from PIL import Image, ImageDraw
from pylab import *
from PCV.clustering import  hcluster
 
def save_vis(filename_list, number_list, pred_pca, output_hcluster, output_PCA):

    imlist = filename_list
    imnbr = len(imlist)

    # Load images, run PCA.
    immatrix = array(pred_pca)
    V, S, immean = pca.pca(immatrix)

    # Project on 2 PCs.
    projected = array([dot(V[[0, 1]], immatrix[i] - immean) for i in range(imnbr)])

    # height and width
    h, w = 1200, 1200
    
    # create a new image with a white background
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # draw axis
    draw.line((0, h/2, w, h/2), fill=(255, 0, 0))
    draw.line((w/2, 0, w/2, h), fill=(255, 0, 0))

    # scale coordinates to fit
    scale = abs(projected).max(0)
    scaled = floor(array([(p/scale) * (w/2 - 20, h/2 - 20) + (w/2, h/2)
                          for p in projected])).astype(int)

    # paste thumbnail of each image
    for i in range(imnbr):
        nodeim = Image.open(imlist[i])
        nodeim.thumbnail((25, 25))
        ns = nodeim.size
        box = (scaled[i][0] - ns[0] // 2, scaled[i][1] - ns[1] // 2,
             scaled[i][0] + ns[0] // 2 + 1, scaled[i][1] + ns[1] // 2 + 1)
        img.paste(nodeim, box)

    tree = hcluster.hcluster(projected)
    hcluster.draw_dendrogram(tree,imlist,filename=output_hcluster)

    for i, num in enumerate(number_list):
        if num < 8:
            color1 = mod(370*num,255)
            color2 = mod(170*num,255)
            color3 = mod(270*num,255)
            draw.text((scaled[i][0],scaled[i][1]),str(num),fill = (color1,color2,color3))
        else:
            draw.text((scaled[i][0],scaled[i][1]),str(num),fill = (255,0,0))

    figure()
    imshow(img)
    axis('off')
    img.save(output_PCA)
    show()