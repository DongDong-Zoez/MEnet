def imshow(album, net_names, loss):
    album = np.transpose(album, [1,2,0])
    plt.figure(figsize=(18,18))
    for i in range(len(loss)):
        plt.text(270+135*i, 16, net_names[i] + ": " + str(round(loss[i], 4)), style='italic',fontweight='bold',
                bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.axis('off')
    plt.imshow(album)

def display_image(net_list, net_names, dataset, loss_func=nn.BCEWithLogitsLoss(), batch_size=1, num_images=8):
    for i in range(num_images):
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
        image, mask = next(iter(dataloader))
        image = image.to(dtype=torch.float32)
        masks = []
        loss = []
        for module in net_list:
            pred = module(image)
            if isinstance(pred, tuple):
                pred = pred[1]
            pred = torch.clamp(pred, 0, 1)
            pred[pred>=0.5] = 1
            pred[pred< 0.5] = 0
            masks.append(pred)
            loss.append(loss_func(pred, mask).item()/batch_size)

        images = []
        img = torch.clone(image)
        img[:,0,:,:][masks[-1][:,0,:,:]==1] = 1
        images.append(img)
        #for m in masks:
        #    img[:,0,:,:][m[:,0,:,:]==1] = 1
        #    images.append(img)
        #    img = torch.clone(image)

        masks = torch.cat(masks, dim=0)
        images = torch.cat(images, dim=0)
        concatenated = torch.cat([image, mask, masks, images], dim=0)
        album = torchvision.utils.make_grid(concatenated, nrow=len(net_list)+3)
        imshow(album, net_names, loss)
