def func(k):
   for i in xrange(k):
        yield (i, 1)
a = func(10)
print a
for i,j in a:
    print i,j
