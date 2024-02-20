Final Project AM170A Inquiry 3: Bacteria Insulin Production

Members:
	-Kevin Silberberg
	-Sebastian Osorio
	-Zoe Laclair

Project Hypothesis Ideas:

we know that the logarithmic model:

2d system of differential equations:

	dB/dt = alpha * B * (1 - B/kappa) - beta * (B*I)

	dI/dt = gamma * B

with initial conditions 

	B(0) > 0
	
	I(0) = 0

where B(t), I(t) are Bacterial population and Insulin concentration at time t.


Idea 1:

Can we model the system by removing the logarithmic term (1 - B / kappa),
by adding a 3rd dimention to the system:

	dS/dt = F(B, I, S) 

where S(t) is the sugar concentration in solution. 
