void matmuld(double **a, double **b, double **c)
{
  for(int i=0;i<1024;i++)
    for(int j=0;j<1024;j++)
      for(int k=0;k<1024;k++)
	c[i][j] += a[i][k]*b[k][j];
}
