function ret=check_gaussian_intersection(mu1,sigma1,mu2,sigma2,rr)

if (nargin<1) test_check_gaussian_intersection; return; end;

% ret=check_gaussian_intersection_old(mu1,sigma1,mu2,sigma2,rr,1000);
% return;

A1=2*inv(sigma1);
b1=-2*inv(sigma1)*mu1;
alpha1=mu1'*inv(sigma1)*mu1-rr^2;

A2=2*inv(sigma2);
b2=-2*inv(sigma2)*mu2;
alpha2=mu2'*inv(sigma2)*mu2-rr^2;

dist0=ellipsoid_distance(A1,b1,alpha1,A2,b2,alpha2);
ret=(dist0==0);

end


function distance=ellipsoid_distance(A1,b1,alpha1,A2,b2,alpha2)
% distance=ellipsoid_distance(A1,b1,alpha1,A2,b2,alpha2)
% Computes the Euclidean distance between two ellipsoids defined by
%   1/2*x'*A1*x + b1'*x + alpha1 <= 0
%   1/2*x'*A2*x + b2'*x + alpha2 <= 0
%
%   A_j is a nxn matrix (positive definite)
%   b_j is a nx1 vector
%   alpha_j is a scalar
%
% Reference: Lin, Anhua, and Shih-Ping Han. "On the distance between
%            two ellipsoids." SIAM Journal on Optimization 13.1 (2002):
%            298-308.

if (nargin<1) test_ellipsoid_distance; return; end;

tol0=1e-5;

% Initiation
c1=-inv(A1)*b1; c2=-inv(A2)*b2;
% For now we use the L_infinity norm for ease
gamma1=0.5/max(abs(A1(:)));
gamma2=0.5/max(abs(A2(:)));

max_iterations=50000;

for it=1:max_iterations
	% Step 1
	for pass=1:2
		if (pass==1)
			cc=c1; dd=c2-c1; AA=A1; bb=b1; aalpha=alpha1;
			sgn=1;
		else
			cc=c1; dd=c2-c1; AA=A2; bb=b2; aalpha=alpha2;
			sgn=-1;
		end;
		qa=1/2*dd'*AA*dd;
		qb=1/2*cc'*AA*dd+1/2*dd'*AA*cc+bb'*dd;
		qc=1/2*cc'*AA*cc+bb'*cc+aalpha;
		discr=qb*qb-4*qa*qc;
		if (discr<0)
			A1
			b1
			alpha1
			A2
			b2
			alpha2
			discr
			it
			warning('Discriminant is less than zero in ellipsoid_distance');
			distance=0; % Is this right?
			return;
		end;
		if (qa==0)
			distance=0; % Is this right? Not in paper. I think it means c1=c2, which i guess means they intersect
			return;
			error('qa is zero in ellipsoid_distance');
		end;
		if (pass==1)
			t1=(-qb+sgn*sqrt(discr))/(2*qa);
		else
			t2=(-qb+sgn*sqrt(discr))/(2*qa);
		end;
	end;

	%Step 2
	if (t2<=t1)
		distance=0;
		return;
	end;
	xbar=c1+t1*(c2-c1);
	ybar=c1+t2*(c2-c1);
	distance=sqrt((xbar-ybar)'*(xbar-ybar));
	if (distance<tol0)
		distance=0;
		return;
	end;

	%Step 3
	theta1=angle_between(ybar-xbar,A1*xbar+b1);
	theta2=angle_between(xbar-ybar,A2*ybar+b2);
	if ((abs(theta1)<tol0)&&(abs(theta2)<tol0))
		return;
	end;

	%Step 4
	c1=xbar-gamma1*(A1*xbar+b1);
	c2=ybar-gamma2*(A2*ybar+b2);
end;

fprintf('distance=%.6f, theta1=%.6f, theta2=%.6f\n',distance,theta1,theta2);
error('Maximum iterations exceeded in ellipsoid_distance');

end

function theta=angle_between(v1,v2)

numer=v1'*v2;
denom=sqrt((v1'*v1)*(v2'*v2));
if denom==0
	theta=0;
	return;
end;
theta=acos(numer/denom);

end