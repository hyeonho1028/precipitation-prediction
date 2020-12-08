import torch
import torch.nn as nn
import torch.nn.functional as F

# class my_rainnet(nn.Module):
#     def __init__(self, config):
#         super(my_rainnet, self).__init__()
#         self.config = config
        
#         # Step 1
#         self.conv1f = nn.Sequential(
#             nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(3,3), padding=(1,1)),
#             # 3*1, 1*3 2개로
#             # nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(3,3), padding=(1,1),  ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#         )
#         self.pool1 = nn.MaxPool2d( kernel_size=(2,2) )

#         # Step 2
#         self.conv2f = nn.Sequential(
#             nn.Conv2d( in_channels = 64, out_channels=128, kernel_size=(3,3), padding=(1,1)  ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#         )
#         self.pool2 = nn.MaxPool2d( kernel_size=(2,2) )

#         # Step 3
#         self.conv3f = nn.Sequential(
#             nn.Conv2d( in_channels = 128, out_channels=256, kernel_size=(3,3), padding=(1,1)  ),
#             nn.ReLU(inplace=True),
#         )
        
#         # Step 4
#         self.upsample4d = nn.ConvTranspose2d(256, 128, kernel_size=(2,2), stride=(2,2), )
#         self.conv4s = nn.Sequential(
#             nn.Conv2d( in_channels = 256, out_channels=128, kernel_size=(3,3), padding=(1,1)  ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#         )

#         # Step 5
#         self.upsample5d = nn.ConvTranspose2d(128, 64, kernel_size=(2,2), stride=(2,2), )
#         self.conv5s = nn.Sequential(
#             nn.Conv2d( in_channels = 128, out_channels=64, kernel_size=(3,3), padding=(1,1)  ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#         )

#         # Step 6
#         self.conv6s = nn.Sequential(
#             nn.Conv2d( in_channels = 64, out_channels=1, kernel_size=(1,1)),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x ):
#         step1 =  self.conv1f( x )
#         step1_pool = self.pool1(step1)
        
#         step2 =  self.conv2f( step1_pool )
#         step2_pool = self.pool1(step2)
        
#         step3 = self.conv3f( step2_pool )

#         step4 = torch.cat([ self.upsample4d( step3 ),  step2 ], dim=1)
#         step5 = self.conv4s( step4 )

#         step6 = torch.cat([ self.upsample5d( step5 ),  step1 ], dim=1)

#         step7 = self.conv5s( step6 )
#         setp8 = self.conv6s( step7 )

#         return setp8
       

class RainNet(nn.Module):
    def __init__(self, config):
        super(RainNet, self).__init__()
        self.config = config
        
        # Step 1
        self.conv1f = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(3,3), padding=(1,1),  ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            
        )
        self.conv1s = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1) ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d( kernel_size=(2,2), stride=(2,2) )

        # Step 2
        self.conv2f = nn.Sequential(
            nn.Conv2d( in_channels = 64, out_channels=128, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv2s = nn.Sequential(
            nn.Conv2d( in_channels = 128, out_channels=128, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d( kernel_size=(2,2) , stride=(2,2) )
        
        # Step 3
        self.conv3f = nn.Sequential(
            nn.Conv2d( in_channels = 128, out_channels=256, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3s = nn.Sequential(
            nn.Conv2d( in_channels = 256, out_channels=256, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d( kernel_size=(2,2) , stride=(2,2) )
        
        # Step 4
        self.conv4f = nn.Sequential(
            nn.Conv2d( in_channels = 256, out_channels=512, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv4s = nn.Sequential(
            nn.Conv2d( in_channels = 512, out_channels=512, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.Sequential(
            nn.Dropout(0.5),
            nn.MaxPool2d( kernel_size=(2,2) , stride=(2,2) )
        )
        
        # Step 5
        self.conv5f = nn.Sequential(
            nn.Conv2d( in_channels = 512, out_channels=1024, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv5s = nn.Sequential(
            nn.Conv2d( in_channels = 1024, out_channels=1024, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Step 6
        # kh code
        # self.upsample6d = nn.ConvTranspose2d(1024, 1024,  kernel_size=2, stride=2)
        self.upsample6d = nn.ConvTranspose2d(1024, 1024,  kernel_size=3, stride=2)
        self.conv6f = nn.Sequential(
            nn.Conv2d( in_channels = 1536, out_channels=512, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv6s = nn.Sequential(
            nn.Conv2d( in_channels = 512, out_channels=512, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Step7
        self.upsample7d = nn.ConvTranspose2d(512, 512,   kernel_size=2, stride=2)
        self.conv7f = nn.Sequential(
            nn.Conv2d( in_channels = 768, out_channels=256, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv7s = nn.Sequential(
            nn.Conv2d( in_channels = 256, out_channels=256, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Step8
        # self.upsample8d = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        # self.conv8f = nn.Sequential(
        #     nn.Conv2d( in_channels = 384, out_channels=128, kernel_size=(3,3), padding=(1,1)  ),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv8s = nn.Sequential(
        #     nn.Conv2d( in_channels = 128, out_channels=128, kernel_size=(3,3), padding=(1,1)  ),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )

        self.upsample8d = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv8f = nn.Sequential(
            nn.Conv2d( in_channels = 384, out_channels=128, kernel_size=(2,2), padding=(1,1)  ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv8s = nn.Sequential(
            nn.Conv2d( in_channels = 128, out_channels=128, kernel_size=(2,2), padding=(1,1)  ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv8d = nn.Sequential(
            nn.Conv2d( in_channels=128, out_channels= 2, kernel_size=(1,1), padding=(2,2)  ),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3,3) )
        )

        
        # Step9
        # self.upsample9d = nn.ConvTranspose2d(128, 128,  kernel_size=2, stride=2)
        # self.conv9f = nn.Sequential(
        #     nn.Conv2d( in_channels = 192, out_channels=64, kernel_size=(3,3), padding=(2,2)  ),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv9s = nn.Sequential(
        #     nn.Conv2d( in_channels = 64, out_channels=64, kernel_size=(3,3), padding=(2,2)  ),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        # self.conv9d = nn.Sequential(
        #     nn.Conv2d( in_channels=64, out_channels= 2, kernel_size=(1,1), padding=(2,2)  ),
        #     nn.BatchNorm2d(2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(9,9) )
        # )
        
    def forward(self, x ):
        step1 =  self.conv1s( self.conv1f( x ) ) 
        step1_pool = self.pool1(step1)
        
        step2 = self.conv2s( self.conv2f( step1_pool ) )
        step2_pool = self.pool2(step2)
        
        step3 = self.conv3s( self.conv3f( step2_pool))
        step3_pool= self.pool3(step3)
        
        step4 =  self.conv4s( self.conv4f( step3_pool)) 
        step4_pool = self.pool4(step4)
        
        step5 =  self.conv5s( self.conv5f( step4_pool)) 

        step6 = torch.cat([ self.upsample6d(step5),   step4 ], dim=1)
        step6 = self.conv6s( self.conv6f( step6)) 
        
        step7 = torch.cat([ self.upsample7d(step6),   step3 ], dim=1)
        step7 = self.conv7s( self.conv7f( step7)) 
        
        step8 = torch.cat([ self.upsample8d(step7),   step2 ], dim=1)
        step8 = self.conv8s( self.conv8f( step8)) 
        step8 = self.conv8d(step8)

        # step9 = torch.cat([ self.upsample9d(step8),   step1 ], dim=1)
        # step9 = self.conv9s( self.conv9f( step9)) 
        # step9 = self.conv9d( step9 )

        return step8






class SmallRainNet(nn.Module):
    def __init__(self, config):
        super(RainNet, self).__init__()
        self.config = config
        
        # Step 1
        self.conv1f = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), padding=(5,5),  ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            
        )
        self.conv1s = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1) ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d( kernel_size=(2,2), stride=(2,2) )

        # Step 2
        self.conv2f = nn.Sequential(
            nn.Conv2d( in_channels = 64, out_channels=128, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv2s = nn.Sequential(
            nn.Conv2d( in_channels = 128, out_channels=128, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Step9
        self.upsample9d = nn.ConvTranspose2d(128, 128,  kernel_size=2, stride=2)
        self.conv9f = nn.Sequential(
            nn.Conv2d( in_channels = 192, out_channels=64, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv9s = nn.Sequential(
            nn.Conv2d( in_channels = 64, out_channels=64, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv9d = nn.Sequential(
            nn.Conv2d( in_channels=64, out_channels= 2, kernel_size=(3,3), padding=(1,1)  ),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(9,9) )
        )
        
class my_rainnet(nn.Module):
    def __init__(self, config):
        super(my_rainnet, self).__init__()
        self.config = config
        
        # Step 1
        self.conv1f = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            # 3*1, 1*3 2개로
            # nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(3,3), padding=(1,1),  ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.pool1 = nn.MaxPool2d( kernel_size=(2,2) )

        # Step 2
        self.conv2f = nn.Sequential(
            nn.Conv2d( in_channels = 64, out_channels=128, kernel_size=(3,3), padding=(1,1)  ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.pool2 = nn.MaxPool2d( kernel_size=(2,2) )

        # Step 3
        self.conv3f = nn.Sequential(
            nn.Conv2d( in_channels = 128, out_channels=256, kernel_size=(3,3), padding=(1,1)  ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )
        self.pool3 = nn.MaxPool2d( kernel_size=(2,2) )
        
        self.conv4f = nn.Sequential(
            nn.Conv2d( in_channels = 256, out_channels=512, kernel_size=(3,3), padding=(1,1)  ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )

        self.upsample5d = nn.ConvTranspose2d(512, 256, kernel_size=(2,2), stride=(2,2), )
        self.conv5s = nn.Sequential(
            nn.Conv2d( in_channels = 512, out_channels=256, kernel_size=(3,3), padding=(1,1)  ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

        # Step 4
        self.upsample6d = nn.ConvTranspose2d(256, 128, kernel_size=(2,2), stride=(2,2), )
        self.conv4s = nn.Sequential(
            nn.Conv2d( in_channels = 256, out_channels=128, kernel_size=(3,3), padding=(1,1)  ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        # Step 5
        self.upsample7d = nn.ConvTranspose2d(128, 64, kernel_size=(2,2), stride=(2,2), )
        self.conv8s = nn.Sequential(
            nn.Conv2d( in_channels = 128, out_channels=64, kernel_size=(3,3), padding=(1,1)  ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )

        # Step 6
        self.conv9s = nn.Sequential(
            nn.Conv2d( in_channels = 64, out_channels=1, kernel_size=(1,1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x ):
        step1 =  self.conv1f( x )
        step1_pool = self.pool1(step1)
        
        step2 =  self.conv2f( step1_pool )
        step2_pool = self.pool1(step2)
        
        step3 = self.conv3f( step2_pool )
        step3_pool = self.pool1(step3)

        step4 = self.conv4f(step3_pool)

        step5 = torch.cat([ self.upsample5d( step4 ),  step3 ], dim=1)
        step6 = self.conv5s( step5 )

        step7 = torch.cat([ self.upsample6d( step6 ),  step2 ], dim=1)
        step8 = self.conv4s( step7 )

        step9 = torch.cat([ self.upsample7d( step8 ),  step1 ], dim=1)

        step9 = self.conv8s( step9 )
        step9 = self.conv9s( step9 )

        return step9