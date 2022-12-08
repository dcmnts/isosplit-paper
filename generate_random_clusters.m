function [clusters,samples,labels]=generate_random_clusters(opts)

if (nargin<1) test_generate_random_clusters; return; end;

if (~isfield(opts,'verbose')) opts.verbose=false; end;
if (~isfield(opts,'nongaussian')) opts.nongaussian=0; end;

ndims=opts.ndims;
zdist=opts.zdist;
pops=opts.pops;
sigma_scale=opts.sigma_scale;
num_clusters=opts.num_clusters;
spread_factor=opts.spread_factor;
anisotropy_factor=opts.anisotropy_factor;

clusters={};

for j=1:num_clusters
	done=false;
	mu=-1+rand(ndims,1)*(2);
	mu=mu/sqrt(mu'*mu);
	min_mu_radius=0;
	max_mu_radius=inf;
	mu_radius=0;
	sigma=random_sigma(ndims,spread_factor,anisotropy_factor)*sigma_scale;
	while ~done	
		if (is_far_enough_away(mu*mu_radius,sigma,clusters,zdist))
			if (mu_radius-min_mu_radius<0.1)
				done=true;
			else
				max_mu_radius=mu_radius;
				mu_radius=(mu_radius+min_mu_radius)/2;
			end;
		else
			min_mu_radius=mu_radius;
			if (mu_radius==0) mu_radius=1;
			else
				mu_radius=min(mu_radius*2,(mu_radius+max_mu_radius)/2);
			end;
		end;
	end;
	clusters{end+1}=struct('sigma',sigma,'mu',mu*mu_radius,'pop',pops(j));
end;

samples=zeros(ndims,0);
labels=zeros(1,0);
for j=1:length(clusters)
	sigma=clusters{j}.sigma;
	mu=clusters{j}.mu;
	pop=clusters{j}.pop;
	samples0=generate_samples(mu,sigma,pop,opts);
	samples=cat(2,samples,samples0);
	labels=cat(2,labels,ones(1,pop)*j);
end;
% N=size(samples,2);
% inds=randsample(N,floor(N*0.1));
% for dd=1:ndims
% 	minval=min(samples(dd,:));
% 	maxval=max(samples(dd,:));
% 	samples(dd,inds)=rand(1,length(inds))*(maxval-minval)+minval;
% end

end

function ret=is_far_enough_away(mu,sigma,clusters,zdist)

for j=1:length(clusters)
	if (check_gaussian_intersection(mu,sigma,clusters{j}.mu,clusters{j}.sigma,zdist))
		ret=false;
		return;
	end;
end;

ret=true;

end

function samples=generate_samples(mu,sigma,num,opts)

n=length(mu);
if ~opts.nongaussian
	samples=randn(n,num);
% 	norms=sqrt(sum(samples.^2,1));
% 	samples=samples./repmat(norms,n,1)*1.9;
else
	samples=randn(n,num);
	for dd=1:n
		tmp=log(abs(randn(1,num)+3));
		tmp=(tmp-mean(tmp))/sqrt(var(tmp))*1.1;
		samples(dd,:)=tmp;
	end;
	R=create_random_rotation(n);
	samples=R*samples;
end;

%norms=sqrt(sum(samples.^2,1));
%samples=samples./repmat(norms,n,1)*1.93;

T=cholcov(inv(sigma));

samples=repmat(mu,1,size(samples,2))+inv(T)*samples;

end

function ret=trnd2(V,n1,n2)

ret=trnd(V,n1,n2);
while true
	inds=find(abs(ret(:))>5);
	if (length(inds)==0)
		return;
	end;
	ret(inds)=trnd(V,1,length(inds));
end;

end

function plot_samples(samples,labels)

colors={'r.','g.','b.','k.','y.','m.','c.','r+','g+','b+','k+','y+','m+','c+'};
for j=1:max(labels)
	xx=samples(1,find(labels==j));
	yy=samples(2,find(labels==j));
	col=colors{mod(j-1,length(colors))+1};
	plot(xx,yy,col);
	if (j==1) hold on; end;
end;
set(gca,'xtick',[],'ytick',[])
sub_pos = get(gca,'position'); % get subplot axis position
sub_pos(1)=sub_pos(1)-sub_pos(3)*0.2;
sub_pos(3)=sub_pos(3)*1.4;
sub_pos(2)=sub_pos(2)-sub_pos(4)*0.2;
sub_pos(4)=sub_pos(4)*1.2;
set(gca,'position',sub_pos) % stretch its width and height
xmin=min(samples(1,:)); xmax=max(samples(1,:));
ymin=min(samples(2,:)); ymax=max(samples(2,:));
xlim([floor(xmin-2),ceil(xmax+2)]);
ylim([floor(ymin-2),ceil(ymax+2)]);

end

function A=random_sigma(n,spread_factor,anisotropy_factor)

A=zeros(n,n);
spread=exp(-(rand()*2-1)*spread_factor);
for j=1:n
	A(j,j)=spread*exp(-(rand()*2-1)*anisotropy_factor);
end;
R=create_random_rotation(n);
A=R*A*R';

%[~,evals]=eig(A);
%evals=diag(evals);
%disp(evals);

end

function R=create_random_rotation(n)

R=eye(n);
for j=1:n-1
	theta=rand()*2*pi;
	RR=eye(n);
	RR(j,j)=cos(theta);
	RR(j,j+1)=sin(theta);
	RR(j+1,j)=-sin(theta);
	RR(j+1,j+1)=cos(theta);
	R=RR*R;
end;

end

function test_generate_random_clusters

close all;

%rng(1);

opts.ndims=2;
opts.zdist=2.5;
opts.num_clusters=8;
opts.pops=randi([100,500],1,opts.num_clusters);
opts.sigma_scale=3;
opts.spread_factor=2;
opts.anisotropy_factor=1.2;
opts.nongaussian=1;
opts.verbose=1;

[clusters,samples,labels]=generate_random_clusters(opts);

plot_samples(samples,labels);

end