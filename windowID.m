function windows = windowID(signalSize, windowSize, overlap) 
% -------------------------------------------------------------------------
% INPUT
%   signalSize
%   windowSize
%   overlap
% OUTPUT
%   windows (rows: window indeces) (columns: number of windows)
%
% NOTE:
%   to keep windowSize in last window, reduced overlap.
%
% EDIT:
% 14-May-2024 by ITHENOA.   {works with samples}
% -------------------------------------------------------------------------

if windowSize < overlap, error("windowSize < overlap"), end
% first window
windows = 1:windowSize;
% main loop
while true
    windowStart = windows(end,end) - overlap + 1;
    windowEnd = windowStart + windowSize - 1;
    % for last window
    if windowEnd(end) > signalSize
        windowEnd = signalSize;
        windowStart = windowEnd - windowSize + 1;
    end
    % concatination
    windows = cat(1, windows, windowStart:windowEnd);
    % stop condition
    if windowEnd(end) == signalSize, break, end
end