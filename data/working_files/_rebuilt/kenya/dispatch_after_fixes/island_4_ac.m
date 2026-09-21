function mpc = island_4_ac
% Fixed-equipment constrained AC dispatch
% Synthetic case; consult accompanying provenance and validation reports.
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
20	3	0	0	0	0	1	1.00060758857501	0	220	1	1.1	0.9;
21	1	8.5521	2.8109	0	0	1	1.00056148279265	-0.00770076562997534	132	1	1.1	0.9;
];
mpc.gen = [
20	8.55211618971557	2.81217898753028	48.72	-5.568	1.00060758857501	100	1	87	0;
];
mpc.branch = [
20	21	2e-05	0.00158	0	1000	1000	1000	0	0	1	-60	60;
];
mpc.gencost = [
2	0	0	3	0.23483	32.9682	1153.248;
];
end
