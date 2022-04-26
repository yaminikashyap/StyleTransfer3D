# training function, read carefully
def train(train_pts_filelist, model, optimizer, args):
    model.train()  # switch to train mode
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    random.shuffle(train_pts_filelist)
    num_batch = np.floor(len(train_pts_filelist) / args.train_batch).astype(int)
    for i in range(num_batch):
        pts_filelist_batch = train_pts_filelist[i*args.train_batch : (i+1)*args.train_batch]
        optimizer.zero_grad()
        for pts_filename in pts_filelist_batch:
            frame_num = int(pts_filename.split("/")[-1].split("_")[-2])
            # load data
            pts = np.load(pts_filename)            
            vtx = np.load(pts_filename.replace("_rpts.npy", "_vtx.npy"))
            corr = np.load(pts_filename.replace("_rpts.npy", "_corr.npy"))
            mask = np.load(pts_filename.replace("_rpts.npy", "_corrmask.npy"))
            # convert to tensor
            pts_tensor = torch.FloatTensor(pts).to(device)
            vtx_tensor = torch.FloatTensor(vtx).to(device)
            corr_tensor = torch.LongTensor(corr).to(device)
            mask_tensor = torch.FloatTensor(mask).to(device)            
            # forward pass
            vtx_feature, pts_feature, pred_mask = model(vtx_tensor, pts_tensor)
            # calculate loss and accuracy
            if args.train_corrmask:  
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_mask.squeeze(dim=1), mask_tensor)
                acc = calc_mask_accuracy(pred_mask.squeeze(dim=1), mask_tensor)
            else:
                loss = NTcrossentropy(vtx_feature, pts_feature, corr_tensor, tau=args.tau_nce)
                acc = calc_matching_accuracy(vtx_feature, pts_feature, corr_tensor, pts_tensor, args.distance_threshold)
            loss.backward()
            loss_meter.update(loss.item())
            acc_meter.update(acc.item())
        # we accumulate gradient for several samples, and make a step by the average gradient
        optimizer.step()
    return loss_meter.avg, acc_meter.avg