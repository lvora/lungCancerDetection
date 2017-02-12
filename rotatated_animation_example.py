import kds17_io as kio
from kds17_pre import *

def animate(image):
    fig = plt.figure()
    ims = []
    ax = fig.add_subplot(111)
    for i in range(image.shape[0]):
        im = plt.imshow(image[i], cmap=plt.cm.gray)
        ims.append([im])
    ani = an.ArtistAnimation(fig, ims, interval=150, blit=True)
    plt.show()

def rotate(image, pos):
    pos_dict = {'anterposterior' : (1,0),
                'r2l_saggital' : (2,0)}
    if pos == 'transverse':
        return image
    else:
        return np.rot90(image,axes=pos_dict[pos])

def main(argv = None):
    io = kio.DicomIO('/home/charlie/kaggle_pickles/')
    a = io.load()
    im = a[0].batch[0].image

    binary_image = np.array(im > -320, dtype=np.int8)+1

    #animate(im) #transverse
    #animate(np.rot90(np.transpose(im,[1,2,0]),axes=(1,2))) #anterposterior
    #animate(np.rot90(np.transpose(im,[2,0,1]),axes=(2,1))) #r2l_saggital

    
    tab_im = rotate(binary_image, 'anterposterior')[::-1]
    


if __name__ == '__main__':
    main()
