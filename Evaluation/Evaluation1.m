clc;
clear;

% all1_t: the result of speaker in frequency domain from my model (concated)
% all2_t: the result of noise in frequency domain from my model (concated)
% source1_t: the result of speaker in frequency domain from my model (speaker by speaker, sentence by sentence)
% source2_t: the result of noise in frequency domain from my model (speaker by speaker, sentence by sentence)
% mix : the mixed signal in time domain
% mix_phase: the phase of mixing signal (speaker by speaker, sentence by sentence)
% source1_test: the result of speaker in time domainfrom my model (speaker by speaker, sentence by sentence)
% source2_test: the result of noise in time domain from my model (speaker by speaker, sentence by sentence)
% test1: the clean speech of speaker in time domain
% test2: the clean noise in time domain
% SNR: the SNR (speaker by speaker, sentence by sentence)
% STOI: the STOI (speaker by speaker, sentence by sentence)

%% Setup

%window setting
%number of fft point
numfft = 1024;
%window size(in second)
win = 0.064;
%overlap size(in second)
shift = 0.032;

time_step  = 20;
batch_size = 50;
time_batch = time_step*batch_size;

%% Loading Data

% mem_dim_mem_size
all1_t = csvread('../Test1/Pred1/pred1_sp_6_cnnntm_test1_mem16.csv');
all2_t = csvread('../Test1/Pred2/pred2_sp_6_cnnntm_test1_mem16.csv');


%% Slice

audpath = '../Test1/Target1/';

source1_t = cell(6, 1);
source2_t = cell(6, 1);

start_idx = 1;
for idx=1:6
    source1_t{idx, 1} = cell(25, 1);
    source2_t{idx, 1} = cell(25, 1);
    sentencePath = strcat(audpath, strcat(int2str(idx), '/'));
    sentenceFile = dir(sentencePath);
    sentenceFile = sentenceFile(3:end);
    for jdx=1:25
        load(strcat(sentencePath, sentenceFile(jdx).name));
        length = size(target1, 2);
        source1_t{idx, 1}{jdx, 1} = all1_t(:, start_idx:start_idx+length-1);
        source2_t{idx, 1}{jdx, 1} = all2_t(:, start_idx:start_idx+length-1);
        start_idx = start_idx+length;
    end
    remain = mod(start_idx, time_batch);
    if remain ~= 0
        start_idx = start_idx + time_batch-remain + 1;
    end
end


%% Read test1 and test2
test1 = cell(6, 1);
test2 = cell(6, 1);
for idx=1:6
    fprintf('Speaker %d\n', idx)
    test1{idx, 1} = cell(25, 1);
    test2{idx, 2} = cell(25, 1);
    audpath1 = strcat('../Test1/Target1/Audio/', strcat(int2str(idx), '/'));
    audpath2 = strcat('../Test1/Target2/Audio/', strcat(int2str(idx), '/'));
    audfiles1 = dir(audpath1); audfiles1 = audfiles1(3:end);
    audfiles2 = dir(audpath2); audfiles2 = audfiles2(3:end);
    for jdx=1:numel(audfiles1)
        [train1, ~ ] = audioread(strcat(audpath1, audfiles1(jdx).name));
        [train2, fs] = audioread(strcat(audpath2, audfiles2(jdx).name));
        train1 = train1./sqrt(sum(train1.^2));
        train2 = train2./sqrt(sum(train2.^2));
        test1{idx, 1}{jdx, 1} = train1';
        test2{idx, 1}{jdx, 1} = train2';
    end
end

%%
% get mix_phase
audpath_pha = '../Test1/Mix/phase/';
audpath_mag = '../Test1/Mix/';

mix_phase = cell(6, 1);
mix_mag   = cell(6, 1);

start_idx=1;
for idx=1:6
    mix_phase{idx, 1} = cell(25, 1);
    mix_mag{idx, 1}   = cell(25, 1);
    sentencePath_pha = strcat(audpath_pha, strcat(int2str(idx), '/'));
    sentenceFile_pha = dir(sentencePath_pha);
    sentenceFile_pha = sentenceFile_pha(3:end);
    sentencePath_mag = strcat(audpath_mag, strcat(int2str(idx), '/'));
    sentenceFile_mag = dir(sentencePath_mag);
    sentenceFile_mag = sentenceFile_mag(3:end);
    for jdx=1:25
        load(strcat(sentencePath_pha, sentenceFile_pha(jdx).name));
        mix_phase{idx, 1}{jdx, 1} = mix_train_phase;
        load(strcat(sentencePath_mag, sentenceFile_mag(jdx).name));
        mix_mag{idx, 1}{jdx, 1}   = mix_train;
        start_idx = start_idx+length;
    end
end

%%
source1_test = cell(6, 1);
source2_test = cell(6, 1);
maxLength    = cell(6, 1);
mix          = cell(6, 1);
clean        = cell(6, 1);
noise        = cell(6, 1);
SNR          = cell(6, 1);
STOI         = cell(6, 1);
STOI_unproc  = cell(6, 1);
for idx=1:6
    source1_test{idx, 1} = cell(25, 1);
    source2_test{idx, 1} = cell(25, 1);
    maxLength{idx, 1}    = cell(25, 1);
    mix{idx, 1}          = cell(25, 1);
    clean{idx, 1}        = cell(25, 1);
    noise{idx, 1}        = cell(25, 1);
    SNR{idx, 1}          = cell(25, 1);
    STOI{idx, 1}         = cell(25, 1);
    for jdx=1:25
        source1_test{idx, 1}{jdx, 1} = OLA(source1_t{idx, 1}{jdx, 1},mix_phase{idx, 1}{jdx, 1},win,shift,fs);
        %source2_test{idx, 1}{jdx, 1} = OLA(source2_t{idx, 1}{jdx, 1},mix_phase{idx, 1}{jdx, 1},win,shift,fs);
        
        mix{idx, 1}{jdx, 1} = OLA(mix_mag{idx, 1}{jdx, 1},mix_phase{idx, 1}{jdx, 1},win,shift,fs);
        
        maxLength{idx, 1}{jdx, 1} = max([size(test1{idx, 1}{jdx, 1}, 2), size(test2{idx, 1}{jdx, 1}, 2)]);
        test1{idx, 1}{jdx, 1}(end+1:maxLength{idx, 1}{jdx, 1})=eps;
        %test2{idx, 1}{jdx, 1}(end+1:maxLength{idx, 1}{jdx, 1})=eps;
        
        test1{idx, 1}{jdx, 1} = test1{idx, 1}{jdx, 1}./sqrt(sum(test1{idx, 1}{jdx, 1}.^2));
        %test2{idx, 1}{jdx, 1} = test2{idx, 1}{jdx, 1}./sqrt(sum(test2{idx, 1}{jdx, 1}.^2));
        
        clean{idx, 1}{jdx, 1} = test1{idx, 1}{jdx, 1}(1:size(source1_test{idx, 1}{jdx, 1}, 2));
        noise{idx, 1}{jdx, 1} = source1_test{idx, 1}{jdx, 1} - clean{idx, 1}{jdx, 1};
        Px1 = sumsqr(clean{idx, 1}{jdx, 1})/size(clean{idx, 1}{jdx, 1}, 2);
        Px2 = sumsqr(noise{idx, 1}{jdx, 1})/size(noise{idx, 1}{jdx, 1}, 2);
        SNR{idx, 1}{jdx, 1}   = log10((Px1-Px2)/Px2);
        STOI{idx, 1}{jdx, 1}  = stoi(test1{idx}{jdx}(1:size(source1_test{idx}{jdx}, 2)), source1_test{idx}{jdx}, fs);
        STOI_unproc{idx, 1}{jdx, 1} = stoi(test1{idx}{jdx}(1:size(mix{idx}{jdx}, 2)), mix{idx}{jdx}, fs);
    end
end


%%
STOI_all = 0;
STOI_unproc_all = 0;
for idx=1:6
    for jdx=1:25        
        STOI_all = STOI_all + STOI{idx}{jdx};
        STOI_unproc_all = STOI_unproc_all + STOI_unproc{idx}{jdx};
    end
end
STOI_all = STOI_all/(6*25)
STOI_unproc_all = STOI_unproc_all/(6*25)

%print(STOI_all);
%print(STOI_unproc_all);
%print(STOI_all - STOI_unproc_all);