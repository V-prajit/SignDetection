import java.util.*;

public class FastDTW {

    // Small helper class that lets us store (i, j) in a Set
    // and compare them by CONTENTS, not by array reference.
    private static class IntPair {
        final int i;
        final int j;

        IntPair(int i, int j) {
            this.i = i;
            this.j = j;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            IntPair that = (IntPair) o;
            return i == that.i && j == that.j;
        }

        @Override
        public int hashCode() {
            return Objects.hash(i, j);
        }
    }

    public static double computeFastDTW(double[][] x, double[][] y, int radius) {
        return fastDTW(x, y, radius, FastDTW::euclideanDistance);
    }

    private static double fastDTW(double[][] x, double[][] y, int radius, DistanceFunction distanceFunction) {
        int minSize = radius + 2;

        if (x.length <= minSize || y.length <= minSize) {
            // If sequences are small, just do a normal DTW
            return dtw(x, y, distanceFunction);
        }

        // Otherwise, reduce the size by half
        double[][] xShrunk = _reduceByHalf(x);
        double[][] yShrunk = _reduceByHalf(y);

        // Recursively get a "low resolution" path
        List<int[]> lowResPath = fastDTWPath(xShrunk, yShrunk, radius, distanceFunction);

        // Expand the path
        Set<IntPair> projectedPath = _expandPath(lowResPath, x.length, y.length);

        // Run DTW only in the expanded window
        return _dtwWithWindow(x, y, projectedPath, radius, distanceFunction);
    }

    // Calculate the "path" only, for the shrunk data
    private static List<int[]> fastDTWPath(double[][] x, double[][] y, int radius, DistanceFunction distanceFunction) {
        if (x.length < 2 || y.length < 2) {
            return new ArrayList<>();
        }
        return _dtwWithWindowPath(x, y, radius, distanceFunction);
    }

    private static double[][] _reduceByHalf(double[][] sequence) {
        int reducedSize = (sequence.length + 1) / 2;
        double[][] reduced = new double[reducedSize][sequence[0].length];

        for (int i = 0; i < reducedSize; i++) {
            for (int j = 0; j < sequence[0].length; j++) {
                if (2 * i + 1 < sequence.length) {
                    reduced[i][j] = (sequence[2 * i][j] + sequence[2 * i + 1][j]) / 2.0;
                } else {
                    reduced[i][j] = sequence[2 * i][j];
                }
            }
        }
        return reduced;
    }

    /**
     * Expand the "low resolution" path into a set of points in the high-resolution matrix.
     */
    private static Set<IntPair> _expandPath(List<int[]> path, int lenX, int lenY) {
        Set<IntPair> expanded = new HashSet<>();
        for (int[] pair : path) {
            int i = pair[0] * 2;
            int j = pair[1] * 2;

            // For each node in the shrunk path, add a small neighborhood in the full matrix
            for (int a = Math.max(0, i - 1); a < Math.min(lenX, i + 2); a++) {
                for (int b = Math.max(0, j - 1); b < Math.min(lenY, j + 2); b++) {
                    expanded.add(new IntPair(a, b));
                }
            }
        }
        return expanded;
    }

    /**
     * DTW with a "window" defined by the projected path. Only cells within that window are computed.
     */
    private static double _dtwWithWindow(double[][] x, double[][] y, Set<IntPair> path, int radius, DistanceFunction distanceFunction) {
        int lenX = x.length, lenY = y.length;
        double[][] cost = new double[lenX][lenY];

        // Fill with infinity
        for (double[] row : cost) {
            Arrays.fill(row, Double.POSITIVE_INFINITY);
        }

        // If path set includes (0,0), initialize cost
        if (path.contains(new IntPair(0, 0))) {
            cost[0][0] = distanceFunction.compute(x[0], y[0]);
        }

        for (int i = 0; i < lenX; i++) {
            for (int j = 0; j < lenY; j++) {
                // If (i,j) is not in the window, skip it
                if (!path.contains(new IntPair(i, j))) {
                    continue;
                }

                double currentDist = distanceFunction.compute(x[i], y[j]);

                if (i > 0 && j > 0) {
                    cost[i][j] = currentDist + Math.min(
                            cost[i - 1][j],
                            Math.min(cost[i][j - 1], cost[i - 1][j - 1])
                    );
                } else if (i > 0) {
                    cost[i][j] = currentDist + cost[i - 1][j];
                } else if (j > 0) {
                    cost[i][j] = currentDist + cost[i][j - 1];
                }
            }
        }
        return cost[lenX - 1][lenY - 1];
    }

    /**
     * Simple "full-window" DTW to create a path for smaller sequences. 
     * This returns a path as a List<int[]> (no change needed).
     */
    private static List<int[]> _dtwWithWindowPath(double[][] x, double[][] y, int radius, DistanceFunction distanceFunction) {
        int lenX = x.length, lenY = y.length;
        double[][] cost = new double[lenX][lenY];

        for (double[] row : cost) {
            Arrays.fill(row, Double.POSITIVE_INFINITY);
        }

        // Quick check: radius = 0 means no window, but let's keep it simple
        if (radius == 0) {
            return new ArrayList<>();
        }

        cost[0][0] = distanceFunction.compute(x[0], y[0]);

        for (int i = 1; i < lenX; i++) {
            cost[i][0] = cost[i - 1][0] + distanceFunction.compute(x[i], y[0]);
        }
        for (int j = 1; j < lenY; j++) {
            cost[0][j] = cost[0][j - 1] + distanceFunction.compute(x[0], y[j]);
        }

        for (int i = 1; i < lenX; i++) {
            for (int j = 1; j < lenY; j++) {
                double currentDist = distanceFunction.compute(x[i], y[j]);
                cost[i][j] = currentDist + Math.min(
                        cost[i - 1][j],
                        Math.min(cost[i][j - 1], cost[i - 1][j - 1])
                );
            }
        }

        // Backtrack to build the path
        return _backtrack(cost);
    }

    /**
     * Simple backtrack that returns a List<int[]> representing the alignment path.
     */
    private static List<int[]> _backtrack(double[][] costMatrix) {
        int i = costMatrix.length - 1, j = costMatrix[0].length - 1;
        List<int[]> path = new ArrayList<>();
        path.add(new int[]{i, j});

        while (i > 0 || j > 0) {
            // Potential moves: up, left, diagonal
            List<int[]> moves = new ArrayList<>();
            if (i > 0) moves.add(new int[]{i - 1, j});
            if (j > 0) moves.add(new int[]{i, j - 1});
            if (i > 0 && j > 0) moves.add(new int[]{i - 1, j - 1});

            if (moves.isEmpty()) break;

            // Pick the move with minimum cost
            int[] bestMove = Collections.min(moves, Comparator.comparingDouble(a -> costMatrix[a[0]][a[1]]));

            i = bestMove[0];
            j = bestMove[1];
            path.add(new int[]{i, j});
        }

        Collections.reverse(path);
        return path;
    }

    /**
     * Full DTW (no window). Fallback if sequences are small or radius is large.
     */
    private static double dtw(double[][] x, double[][] y, DistanceFunction distanceFunction) {
        int lenX = x.length, lenY = y.length;
        double[][] cost = new double[lenX][lenY];

        for (double[] row : cost) {
            Arrays.fill(row, Double.POSITIVE_INFINITY);
        }

        cost[0][0] = distanceFunction.compute(x[0], y[0]);

        for (int i = 1; i < lenX; i++) {
            cost[i][0] = cost[i - 1][0] + distanceFunction.compute(x[i], y[0]);
        }
        for (int j = 1; j < lenY; j++) {
            cost[0][j] = cost[0][j - 1] + distanceFunction.compute(x[0], y[j]);
        }

        for (int i = 1; i < lenX; i++) {
            for (int j = 1; j < lenY; j++) {
                double currentDist = distanceFunction.compute(x[i], y[j]);
                cost[i][j] = currentDist + Math.min(
                        cost[i - 1][j],
                        Math.min(cost[i][j - 1], cost[i - 1][j - 1])
                );
            }
        }
        return cost[lenX - 1][lenY - 1];
    }

    private static double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    @FunctionalInterface
    interface DistanceFunction {
        double compute(double[] a, double[] b);
    }
}
