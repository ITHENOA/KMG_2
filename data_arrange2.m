clear;clc

% ADJUSTBLE PARAMETERS
windowSize = 150;
overlap = 140;
% incrimental = 10; ?
% imgSize = [28 28];

% FIX PARAMETERS
classes = categorical(1:11);

% CREATE FOLDER
parentFolder = "data\win150_ov140";
for i = 1:11
    filename = fullfile(parentFolder,string(classes(i)));
    if ~isfolder(filename)
        mkdir(filename);
    end
end

% LOAD SIGNAL
sig = load("data\agregated11.mat").agregated11;

%% MAIN LOOP
% for i = 1:windowSize:size(sig,1)

% first window
windowsIdx = windowID(size(sig,1), windowSize, overlap);
i = 0;
for window = windowsIdx'
    % img = [];
    % for sensor = 1:12
    %     img = cat(3, img, scalogram(sig(window, sensor), imgSize));
    % end
    img = sig(window,1:end-1);

    % save
    className = mode(sig(window,end));
    currentFolder = (fullfile(parentFolder,string(className)));
    newName = getNewName(currentFolder,className,"mat");
    save(fullfile(currentFolder,newName),"img")

    % progress bar
    i = i + 1;
    pbar(i,size(windowsIdx,1), "Info",'data arranging ...')
end

%% FUNCTIONS
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