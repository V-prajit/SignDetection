import py4j.GatewayServer;

public class DTWServer {
    public double calculateDTW(double[][] seq1, double[][] seq2) {
        return DTW.calculateDTW(seq1, seq2);
    }

    public double computeFastDTW(double[][] seq1, double[][] seq2, int radius) {
        return FastDTW.computeFastDTW(seq1, seq2, radius);
    }

    public static void main(String[] args) {
        DTWServer server = new DTWServer();
        GatewayServer gatewayServer = new GatewayServer(server);
        gatewayServer.start();
        System.out.println("DTW Server Started...");
    }
}
