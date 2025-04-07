import py4j.GatewayServer;
import java.util.concurrent.*;
import java.util.*;

public class DTWServer {
    private final ExecutorService threadPool;
    
    public DTWServer() {
        // Create thread pool with core count
        int processors = Runtime.getRuntime().availableProcessors();
        threadPool = Executors.newFixedThreadPool(processors);
        System.out.println("DTW Server initialized with " + processors + " threads");
    }
    
    // Original single calculation method
    public double calculateDTW(double[][] seq1, double[][] seq2) {
        return DTW.calculateDTW(seq1, seq2);
    }

    // Original FastDTW method
    public double computeFastDTW(double[][] seq1, double[][] seq2, int radius) {
        return FastDTW.computeFastDTW(seq1, seq2, radius);
    }

    // NEW BATCH METHOD: Process all comparisons in one call
    // This version accepts a List of sequences rather than a 3D array
    public double[] batchCalculateDTW(double[][] querySequence, List<double[][]> databaseSequences) {
        int totalSequences = databaseSequences.size();
        double[] results = new double[totalSequences];
        
        // Create tasks for parallel execution
        List<Future<DTWResult>> futures = new ArrayList<>(totalSequences);
        
        for (int i = 0; i < totalSequences; i++) {
            final int index = i;
            futures.add(threadPool.submit(() -> {
                double distance = DTW.calculateDTW(querySequence, databaseSequences.get(index));
                return new DTWResult(index, distance);
            }));
        }
        
        // Collect results
        for (Future<DTWResult> future : futures) {
            try {
                DTWResult result = future.get();
                results[result.index] = result.distance;
            } catch (Exception e) {
                System.err.println("Error in DTW calculation: " + e);
            }
        }
        
        return results;
    }
    
    // Helper class for thread results
    private static class DTWResult {
        final int index;
        final double distance;
        
        DTWResult(int index, double distance) {
            this.index = index;
            this.distance = distance;
        }
    }
    
    // Shutdown thread pool
    public void shutdown() {
        threadPool.shutdown();
    }

    public static void main(String[] args) {
        DTWServer server = new DTWServer();
        
        // Simple GatewayServer initialization
        GatewayServer gatewayServer = new GatewayServer(server);
        gatewayServer.start();
        
        System.out.println("DTW Server Started");
        
        // Add shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            server.shutdown();
        }));
    }
}