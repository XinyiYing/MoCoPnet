function [LSNRG,sigma_b,SCR,CON]=Value(tarImg,m,n,d,tar_x,tar_y)%img_IPI,m,n,d,str2num(x1{j+1})*scale,str2num(y1{j+1})*scale
%tarImg 原始目标图像
% 目标真实size b*a
%d 邻域背景宽度
% tar_x，tar_y 目标中心坐标
tarImg = double(tarImg);
[width, height] = size(tarImg);
x_initial=tar_x-floor(m/2);
y_initial=tar_y-floor(n/2);
x_end = tar_x+floor(m/2);
y_end = tar_y+floor(n/2);
if x_initial<1
    x_initial=1;
end
if y_initial<1
    y_initial=1;
end
if x_end>width
    x_end = width;
end
if y_end>height
    y_end = height;
end
target_area = tarImg(x_initial:x_end,y_initial:y_end);

tarImg1 = tarImg;
tarImg1(x_initial:x_end,y_initial:y_end)=-10000;
x_initial1=tar_x-floor(m/2)-d;
y_initial1=tar_y-floor(n/2)-d;
x_end1 = tar_x+floor(m/2)+d;
y_end1 = tar_y+floor(n/2)+d;
if x_initial1<1
    x_initial1=1;
end
if y_initial1<1
    y_initial1=1;
end
if x_end1>width
    x_end1 = width;
end
if y_end1>height
    y_end1 = height;
end
background_area = tarImg1(x_initial1:x_end1,y_initial1:y_end1);
mask111 = find(background_area~=-10000);
background_area1 = background_area(mask111);
temp_max = max(background_area1);
temp_std = std2(background_area1);
if temp_max == 0
    temp_max = 1e-10;
end
if temp_std == 0
    temp_std = 1e-10;
end
LSNRG=double(max(target_area(:)))/double(temp_max);
sigma_b = temp_std;
SCR = abs(mean(target_area(:))-mean(background_area1))/temp_std;
CON = abs(mean(target_area(:))-mean(background_area1));
