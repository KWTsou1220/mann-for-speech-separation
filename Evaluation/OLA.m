function y=OLA(mag,phase,window,shift,fs)
%
%       Overlap and Add(OLA_LSE)  
%       
%       Ref. Thomas F. Quatieri "Discrete-Time Speech Signal Processing Principles And Practice" 
%
%       C.-C. Hsu
%       v1.00: 08-Nov.2011
spec=max(eps,mag).*exp(phase*sqrt(-1));
w_l=fs*window;
s_l=fs*shift;
speech_l=w_l+(size(mag,2)-1)*s_l;
y=zeros(1,speech_l);
W=zeros(1,speech_l);
for n=1:size(mag,2)
    temp=real(ifft([spec( :, n);conj(flipud(spec( 2:size(mag,1)-1, n)))])');
    temp(1:w_l)=temp(1:w_l).*hamming(w_l)';
    y((n-1)*s_l + 1 : (n-1)*s_l+ w_l)=y((n-1)*s_l + 1 : (n-1)*s_l+ w_l)+temp(1:w_l);
    W((n-1)*s_l + 1 : (n-1)*s_l+ w_l)=W((n-1)*s_l + 1 : (n-1)*s_l+ w_l)+(hamming(w_l).^2)';
end
%y=y./W;
%y=y(:);