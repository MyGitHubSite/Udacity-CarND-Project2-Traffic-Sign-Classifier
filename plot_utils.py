import matplotlib.pyplot as plt
import os.path

# plot histogram of labels
def plot_histogram(x_data, y_data, title, save_path=''):
    plt.figure(figsize=(12, 4))
    plt.bar(x_data, y_data)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    plt.savefig(os.path.join(save_path + title + '.png'))
    plt.show()
    return None

# plot list of images in subplots
def plot_images(images, labels, cols=1):
    rows = len(images)//cols
    f = plt.figure(figsize=(12,24))
    for i, image in enumerate(images):
        ax = f.add_subplot(rows, cols, i+1)
        ax.set_title(labels[i])   #+str(image.shape))
        plt.imshow(image, cmap='gray')
    plt.tight_layout()
    plt.show()
    return None

# plot list of images in subplots
def plot_images2(index, images, labels, cols=1, title='', save_path=''):
    rows = len(images)//cols
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12,2), dpi=80)
    for i, image in enumerate(images):
        ax = plt.subplot(rows, cols, i+1)
        ax.set_xticks([0,8,16,24,32])
        ax.set_yticks([0,8,16,24,32])
        ax.set_axis_off()
        ax.imshow(image)

    #plt.tight_layout()
    fig.suptitle(title)
    fig.subplots_adjust(top=0.9, bottom=0.0)
    plt.savefig(os.path.join(save_path + title + '.png'))
    plt.show()
    return None

# plot list of images in subplots
def plot_images3(index, images, labels, sub_labels, cols=1, title='', save_path=''):
    rows = len(images)//cols
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12,2), dpi=80)
    for i, image in enumerate(images):
        ax = plt.subplot(rows, cols, i+1)
        ax.set_xticks([0,8,16,24,32])
        ax.set_yticks([0,8,16,24,32])
        ax.set_axis_off()
        ax.set_title(sub_labels[i])
        ax.imshow(image, cmap='gray')

    #plt.tight_layout()
    fig.suptitle(title)
    fig.subplots_adjust(top=0.9, bottom=0.0)
    plt.savefig(os.path.join(save_path + title + '_processed.png'))
    plt.show() 
    return None   

#Plot and Save Training and Validation Accuracy
def plot_model_results(i, l_epochs, l_train_acc, l_valid_acc, l_test_acc, y_min=0.1, y_max=1, title='', save_path='', show=False):
    plt.plot(l_epochs, l_train_acc, 'r--', label='Training={0:.3f}'.format(l_train_acc[-1]))
    plt.plot(l_epochs, l_valid_acc,'b--', label='Validation={0:.3f}'.format(l_valid_acc[-1]))
    plt.plot(l_epochs, l_test_acc*len(l_epochs),'g--', label='Test={0:.3f}'.format(l_test_acc[-1]))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.axis([1, len(l_epochs)+1, y_min, y_max])
    plt.legend(loc='lower right')
    plt.title("Training, Validation, and Test Accuracy")
    plt.savefig(os.path.join(save_path + title + '.png'))
    if show==False: 
    	plt.close()
    else:
    	plt.show()
    return None

#Plot and Save Validation Accuracy by Class
def plot_class_results(i, l_class, l_acc, t_acc, y_min=0.1, y_max=1, title='', save_path='', show=False):
    plt.figure(figsize=(12, 4))
    plt.bar(l_class, l_acc, 0.6, label='Class Accuracy')
    plt.plot(l_class, t_acc*len(l_class),'r--', label='Total Accuracy={:.3f}'.format(t_acc[0]))
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.axis([-1, len(l_class), 0.4, 1])
    plt.legend(loc='lower center')
    plt.xticks(l_class)
    plt.savefig(os.path.join(save_path + title + '.png'))
    if show==False: 
    	plt.close()
    else:
    	plt.show()
    return None


