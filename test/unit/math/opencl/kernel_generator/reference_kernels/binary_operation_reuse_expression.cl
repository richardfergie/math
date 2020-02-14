kernel void calculate(__global double* var4_global, int var4_rows, int var4_view, __global double* var5_global, int var5_rows, int var5_view, __global double* var8_global, int var8_rows, int var8_view, __global double* var9_global, int var9_rows, int var9_view, const int rows, const int cols){
int gid_i = get_global_id(0);
int gid_j = get_global_id(1);
int gsize_i = get_global_size(0);
int gsize_j = get_global_size(1);
for(int j = gid_j; j < cols; j += gsize_j){
for(int i = gid_i; i < rows; i += gsize_i){
double var4 = 0; if (!((!contains_nonzero(var4_view, LOWER) && j < i) || (!contains_nonzero(var4_view, UPPER) && j > i))) {var4 = var4_global[i + var4_rows * j];}
double var5 = 0; if (!((!contains_nonzero(var5_view, LOWER) && j < i) || (!contains_nonzero(var5_view, UPPER) && j > i))) {var5 = var5_global[i + var5_rows * j];}
double var3 = var4 + var5;
double var7 = var3 * var3;
double var8 = 0; if (!((!contains_nonzero(var8_view, LOWER) && j < i) || (!contains_nonzero(var8_view, UPPER) && j > i))) {var8 = var8_global[i + var8_rows * j];}
double var6 = var7 / var8;
double var2 = var3 / var6;
double var1 = var2 * var6;
var9_global[i + var9_rows * j] = var1;
}
}
}
