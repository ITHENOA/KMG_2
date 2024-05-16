clear;clc
for folder = ["12-Sep-2022-Test-Day-1-Evening-Layout3",...
        "12-Sep-2022-Test-Day-1-Morning-Layout2",...
        "13-Sep-2022-Test-Day-2-Morning-Layout1"]

% MAKE FOLDERS
classes = categorical(["fist","fist0","index","index0","middle","middle0","pinky","pinky0",...
    "ring","ring0","thumb","thumb0","tripod","tripod0","wristUp","wristUp0",...
    "wristUpFist","wristUpFist0","wristUpThumb","wristUpThumb0",...
    "wristUpTripod","wristUpTripod0"]);
matFilesDir = dir(fullfile(folder,"*.mat"));
parentFolder = "data_mat_28_mag2";
for i = 1:length(classes)
    filename = fullfile(parentFolder,string(classes(i)));
    if ~isfolder(filename)
        mkdir(filename);
    end
end

% PARAMETERS
imgSize = [28 28];
nChannel = 16;
Magnitude = @(x) sqrt(x(:,1).^2 + x(:,2).^2 + x(:,3).^2);

% MAIN
nFiles = length(matFilesDir);
for file = 1:nFiles
    gest = extractBefore(matFilesDir(file).name,",");
    keyGest = get_keyGest(gest);
    [sig,idx] = findLabel(fullfile(folder,matFilesDir(file).name), keyGest);
    Image = [];
    for i = 1:length(idx)-1
        if rem(i,2)==0
            cls_name = gest+"0";
        else
            cls_name = gest;
        end
        Image = [];
        for ch = 1:nChannel
            sig_mag = Magnitude(sig(idx(i):idx(i+1)-1, 1+(ch-1)*3:ch*3));
            Image = cat(3, Image, scalogram(sig_mag, imgSize));
        end
        currentFolder = fullfile(parentFolder,cls_name);
        imgName = getNewName(currentFolder,cls_name,"mat");
        % save(fullfile(currentFolder,imgName),"Image")
    end
    progressBar(file,nFiles)
end
end
%%
clear;clc
Magnitude = @(x) sqrt(x(:,1).^2 + x(:,2).^2 + x(:,3).^2);
agregated11 = [];
for folder = ["12-Sep-2022-Test-Day-1-Evening-Layout3",...
        "12-Sep-2022-Test-Day-1-Morning-Layout2",...
        "13-Sep-2022-Test-Day-2-Morning-Layout1"]
    matFilesDir = dir(fullfile(folder,"*.mat"));
    nFiles = length(matFilesDir);
    for file = 1:nFiles
        sig = load(fullfile(folder,matFilesDir(file).name)).sensorsDataCalibratedFiltered;
        gest = extractBefore(matFilesDir(file).name,",");
        clsName = get_GestID(gest);
        sig_mag = [];
        for ch = 1:16
            sig_mag = cat(2, sig_mag, Magnitude(sig(:, 1+(ch-1)*3:ch*3)));
        end
        sig_mag(:,17) = deal(clsName);
        agregated11 = cat(1, agregated11, sig_mag);
        pbar(file,nFiles)
    end
end
%% FUNCTIONS
function [sig,idx] = findLabel(fileName,gest)
    key = load(fileName).(gest);
    sig = load(fileName).sensorsDataCalibratedFiltered;
    [~, idx_max] = findpeaks(key);
    [~, idx_min] = findpeaks(-key);
    idx = [1 idx_max idx_min length(key)];
    idx = sort(idx);
    plot(normalize(key))
    hold on 
    Magnitude = @(x) sqrt(x(:,1).^2 + x(:,2).^2 + x(:,3).^2);
    plot(normalize(Magnitude(sig(:,1:3))))
    for i = 1:length(idx)-1
        if (idx(i+1)-idx(i)) < 10
            if idx(i+1) == idx(end)
                idx(end) = [];
                warning("idx end removed")
            else
                error("vasta")
            end
        end
    end
end
% -----------------------------------
function img = scalogram(data,imgSize)
    fb = cwtfilterbank('SignalLength',length(data),'VoicesPerOctave',12);
    cfs = abs(fb.wt(data));
    img = rescale(cfs); % between [0,1]
    % img = im2uint8(img); % not recommended for .mat (.jpg?)
    img = imresize(img, imgSize); % resize
end
% -----------------------------------
function newImgName = getNewName(currentFolder,mainName,type)
    clasDir = dir(currentFolder);
    newName = numel({clasDir.name}) - 2;
    newImgName = string(mainName)+"_"+newName+"."+string(type);
end
% -----------------------------------
function testDataset(parentFolder,targetSize)
    nImg = [];
    clasDir = {dir(parentFolder).name};
    classes = categorical({clasDir{3:end}});
    for i = 3:numel(clasDir)
        imgFolder = fullfile(parentFolder,clasDir{i});
        imgNames = {dir(imgFolder).name};
        nImg = cat(2,nImg,numel(imgNames));
        for j = 1:numel(imgNames)
            imgDir = fullfile(imgFolder,imgNames{i});
            img = load(imgDir).Image;
            imgSize = size(img);
            if sum(imgSize ~= targetSize)
                warning(['Size : ',imgNames{i}])
            end
            if sum(isnan(img))
                warning(['NAN : ',imgNames{i}])
            end
        end
        progressBar(i,numel(clasDir))
    end
    disp('Test Done')
    histogram('Categories',classes,'BinCounts',nImg)
end
% -----------------------------------
function keyGest = get_keyGest(gest)
    switch gest
        case 'fist'; keyGest = "ring";
        case 'index'; keyGest = "index";
        case 'middle'; keyGest = "middle";
        case 'pinky'; keyGest = "pinky";
        case 'ring'; keyGest = "ring";
        case 'thumb'; keyGest = "thumb";
        case 'tripod'; keyGest = "ring";
        case 'wristUp'; keyGest = "wrist";
        case 'wristUp-fist'; keyGest = "ring";
        case 'wristUp-thumb'; keyGest = "thumb";
        case 'wristUp-tripod'; keyGest = "ring";
        otherwise; error('Invalid gest name')
    end
end
% -----------------------------------
function keyGest = get_GestID(gest)
    switch gest
        case 'fist'; keyGest = 1;
        case 'index'; keyGest = 2;
        case 'middle'; keyGest = 3;
        case 'pinky'; keyGest = 4;
        case 'ring'; keyGest = 5;
        case 'thumb'; keyGest = 6;
        case 'tripod'; keyGest = 7;
        case 'wristUp'; keyGest = 8;
        case 'wristUp-fist'; keyGest = 9;
        case 'wristUp-thumb'; keyGest = 10;
        case 'wristUp-tripod'; keyGest = 11;
        otherwise; error('Invalid gest name')
    end
end