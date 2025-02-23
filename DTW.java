import java.util.Arrays;

public class DTW {
    public static double calculateDTW(double[][] seq1, double[][] seq2) {
        int n = seq1.length;
        int m = seq2.length;
        double[][] dtw = new double[n + 1][m + 1];

        for (double[] row : dtw)
            Arrays.fill(row, Double.POSITIVE_INFINITY);
        dtw[0][0] = 0;

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                double cost = euclideanDistance(seq1[i - 1], seq2[j - 1]);
                dtw[i][j] = cost + Math.min(dtw[i - 1][j], Math.min(dtw[i][j - 1], dtw[i - 1][j - 1]));
            }
        }
        return dtw[n][m];    
    }

    private static double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
}