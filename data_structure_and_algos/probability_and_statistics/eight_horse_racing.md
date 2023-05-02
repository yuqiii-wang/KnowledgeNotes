# Eight Horse Racing

How many competitions should be performed to find the top 8 horses among 64 horses 
(in each competition, only the horse rankings are recorded) ?

$8$ tests: Group the $64$ horses into $8$ groups, and perform $8$ competitions for all $64$ horses.
The top $8$ horses should be all in one group, or different groups.

The $9$-th test: conduct a competition for all top one horses from the $8$ groups.
The top one horse in this competition must also be the top one in the whole $64$ horses;
The second top horse in this test should compete with the second top horse in the top one horse group for the second top place in the whole $64$ horses;
The third top horse should compete the third top horse in the top one horse group, as well as the second top horse in the second top horse group for the third top place, and so on.

The $10$-th test: conduct a competition for the top two, three, four in the top horse group;
top one, two, three in the the second horse group, and the third top horse from the $9$-th test.
This test should find the second top, third top horses from the all $64$ horses.

The $11$-th test: 