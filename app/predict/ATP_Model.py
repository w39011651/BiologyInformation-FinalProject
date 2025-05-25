import collections.abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

class ATPBindingCNN(nn.Module):
    def __init__(self, filter_lengths=[3, 5, 7, 9, 11, 13], num_filters_per_length=32, fc_hidden_dim=128, num_classes=2, dropout_rate = 0.5):
        """
        Args:
            filter_lengths (int or list): 卷積濾波器的長度(高度)。
                                         如果為 int，表示使用單一長度的濾波器。
                                         如果為 list，表示使用列表中所有長度的濾波器。
                                         寬度固定為 20 (PSSM 的寬度)。
                                         所有長度應 <= 15 (PSSM 的高度)。
            num_filters_per_length (int): 每種長度的濾波器數量 (即輸出通道數)。
            fc_hidden_dim (int): 全連接層的隱藏單元數量。
            dropout_rate (float): Dropout 概率，介於 0 到 1 之間。
            num_classes (int): 輸出類別數量 (ATP-binding 或 Non-ATP-binding)。
        """
        super(ATPBindingCNN, self).__init__()

        if isinstance(filter_lengths, int):
            self.filter_lengths = [filter_lengths]
        elif isinstance(filter_lengths, collections.abc.Iterable):
            self.filter_lengths = list(filter_lengths)
        else:
            raise TypeError("filter_lengths must be iterable")
        
        for f_len in self.filter_lengths:
            if f_len <= 0 or f_len > 15:
                raise ValueError(f"filter_lengths must greater than 0 and smaller than or equal to the height of the input matrix")
        
        self.num_filters_per_length = num_filters_per_length
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        ####Convolutional Layer####
        #使用nn.ModuleList存儲不同kernel_size的卷積層
        self.conv_layers = nn.ModuleList()
        self.conv_bns = nn.ModuleList()
        for f_len in self.filter_lengths:
            #The shape of the Kernel Size = (height, width)
            #Input Channel = 1 (PSSM can be considered as a one-channel image)
            #Output Channel = num_filters_per_length
            #Stride = 1
            #Without padding
            conv_layer = nn.Conv2d(in_channels=1,
                                   out_channels=num_filters_per_length,
                                   kernel_size=(f_len, 20),
                                   stride=1,
                                   padding=0)
            self.conv_layers.append(conv_layer)
            self.conv_bns.append(nn.BatchNorm2d(num_filters_per_length))
        ####Convolutional Layer####

        ####Pooling Layer####
        # 計算經過卷積和 One-Max Pooling 後的總特徵數量
        # 每個卷積層會輸出 num_filters_per_length 個特徵圖
        # 經過 One-Max Pooling 後，每個特徵圖會被縮減為一個單一值
        # 總特徵數 = 使用的濾波器長度種類數量 * 每種長度的濾波器數量
        total_pooled_features = len(self.filter_lengths) * num_filters_per_length
        ####Pooling Layer####

        ####Fully Connected Layer####
        # 輸入維度是所有 One-Max Pooled 特徵的總數
        self.fc1 = nn.Linear(in_features=total_pooled_features,
                             out_features=fc_hidden_dim)
        ####Fully Connected Layer####

        self.fc1_bn = nn.BatchNorm1d(fc_hidden_dim)

        ####Dropout Layer####
        self.dropout = nn.Dropout(p=dropout_rate)
        ####Dropout Layer####

        ####Output Layer####
        # 輸入維度是前一個全連接層的輸出維度
        self.fc_out = nn.Linear(in_features=fc_hidden_dim,
                                out_features=num_classes)
        ####Output Layer####

        #ReLU Activate Function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        模型的前向傳播。

        Args:
            x (torch.Tensor): 輸入的 PSSM 窗口批次。
                              期望形狀為 (Batch_Size, 1, Height, Width)
                              對於您的數據，Height=15, Width=20
                              即 (Batch_Size, 1, 15, 20)。

        Returns:
            torch.Tensor: 模型的原始輸出 (logits)，形狀為 (Batch_Size, num_classes)。
        """
        # 確保輸入張量形狀正確 (Batch_Size, 1, 15, 20)
        # 如果您的 DataLoader 回傳的是 (Batch_Size, 15, 20)，您需要在這裡增加通道維度
        if x.dim() == 3:
            # 增加通道維度，形狀變為 (Batch_Size, 1, 15, 20)
            x = x.unsqueeze(1)
        elif x.dim() != 4 or x.shape[1] != 1 or x.shape[2] != 15 or x.shape[3] != 20:
             raise ValueError(f"輸入張量形狀不正確，期望 (Batch_Size, 1, 15, 20)，但收到 {x.shape}")
        
        pooled_outputs = []#Store the result of all dirrerent length filter One-Max Pooling

        for i, conv_layer in enumerate(self.conv_layers):
            # 應用卷積層
            # conv_layer(x) 的輸出形狀為 (Batch_Size, num_filters_per_length, Output_Height, Output_Width)
            # 對於 kernel_size=(f_len, 20), stride=1, padding=0:
            # Output_Height = (15 - f_len)/1 + 1 = 15 - f_len + 1
            # Output_Width = (20 - 20)/1 + 1 = 1
            conv_out = conv_layer(x)

            # 應用BatchNorm2d
            conv_out = self.conv_bns[i](conv_out)

            # 應用ReLU激活
            conv_out = self.relu(conv_out)

            # 應用 One-Max Pooling
            # 論文描述 One-Max Pooling 取 (N-fi+1) 激活值中的最大值
            # N=15, fi 是濾波器長度。N-fi+1 = 15 - fi + 1 是卷積輸出的高度
            # 我們需要在卷積輸出高度維度 (dim=2) 上取最大值
            # F.max_pool2d 的 kernel_size 應該覆蓋整個高度 (conv_out.shape[2])
            # 步長 stride 設為 1
            # 輸出形狀將是 (Batch_Size, num_filters_per_length, 1, 1)
            one_max_pooled = F.max_pool2d(conv_out,
                                          kernel_size=(conv_out.shape[2], 1),# kernel_size 覆蓋整個 Output_Height
                                          stride=1)
            # 將池化後的結果展平
            # 從 (Batch_Size, num_filters_per_length, 1, 1) 變為 (Batch_Size, num_filters_per_length)
            pooled_outputs.append(one_max_pooled.view(x.size(0), -1))

        # 將所有不同長度濾波器的 One-Max Pooled 輸出連接起來
        # cat_dim=1 表示在特徵維度上連接
        all_pooled_features = torch.cat(pooled_outputs, dim=1)
        # 連接後的形狀為 (Batch_Size, total_pooled_features)

        # 應用第一個全連接層
        fc_out = self.fc1(all_pooled_features)
        # 應用第一個全連接層的BatchNorm
        fc_out = self.fc1_bn(fc_out)
        # 應用 ReLU 激活
        fc_out = self.relu(fc_out)

        # 應用 Dropout 層
        # Dropout 只在訓練模式 (model.train()) 時起作用
        fc_out = self.dropout(fc_out)

        # 應用輸出層，獲取最終的 logits
        logits = self.fc_out(fc_out)
        # 輸出 logits 形狀為 (Batch_Size, num_classes)
        return logits


