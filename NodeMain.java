import java.io.File;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;

public class NodeMain {

    static String mCurModelFile = "0.mdl";

    private static final String MODEL_FILE_SUFFIX = ".mdl";
    private static final String BROADCAST_ADDRESS = "255.255.255.255";
    private static final int BROADCAST_RECEIVER_PORT = 50000;
    private static final int BROADCAST_INTERVAL_MSEC = 1000;

    private static final String CMD_REQUEST_MODEL = "CMD_REQUEST_MODEL";

    public static void main(String[] args) {

        refreshCurrentModelFile();

        Thread broadcastReceiverThread = new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("BroadcastReceiver started.");
                DatagramSocket receiverSocket = null;
                try {
                    receiverSocket = new DatagramSocket(BROADCAST_RECEIVER_PORT);
                    byte[] buf = new byte[256];
                    while (true) {
                        DatagramPacket packet = new DatagramPacket(buf, buf.length);
                        receiverSocket.receive(packet);
                        String receivedMsg = new String(packet.getData(), 0, packet.getLength());

                        // TODO parse and verify broadcast signature

                        // discard self broadcast
                        if (mCurModelFile.equals(receivedMsg)) {
                            continue;
                        }

                        // case #1 request a model if peer model is newer than mine
                        if (getModelTimeStamp(receivedMsg) > getModelTimeStamp(mCurModelFile)) {
                            // TODO open file receiver thread
                            // TODO send file receiver info with CMD
                        }

                        // case #2 response
                        if (receivedMsg.equals(CMD_REQUEST_MODEL)) {
                            // TODO open file sender thread
                        }
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                } finally {
                    if (receiverSocket != null) {
                        receiverSocket.close();
                    }
                }
            }

        });
        broadcastReceiverThread.start();

        // send broadcast with an interval
        DatagramSocket senderSocket = null;
        while(true) {
            try {
                Thread.sleep(BROADCAST_INTERVAL_MSEC); // interval
                refreshCurrentModelFile(); // refresh
                senderSocket = new DatagramSocket();
                senderSocket.setBroadcast(true);
                DatagramPacket packet = new DatagramPacket(mCurModelFile.getBytes(),
                        mCurModelFile.getBytes().length, InetAddress.getByName(BROADCAST_ADDRESS),
                        BROADCAST_RECEIVER_PORT);
                senderSocket.send(packet);

            } catch (InterruptedException | IOException e) {
                e.printStackTrace();
            } finally {
                if (senderSocket != null) {
                    senderSocket.close();
                }
            }
        }
    }

    public static void refreshCurrentModelFile() {
        // read current model name
        String path = System.getProperty("user.dir");
        File curDir = new File(path);
        for (File file : curDir.listFiles()) {
            String curFileName = file.getName();
            if (curFileName.endsWith(MODEL_FILE_SUFFIX)) {
                mCurModelFile = Math.max(getModelTimeStamp(curFileName),
                        getModelTimeStamp(mCurModelFile)) + MODEL_FILE_SUFFIX;
            }
        }
    }

    // Note. model name should be "yymmddhhmmss.mdl"
    public static int getModelTimeStamp(String fileName) {
        return Integer.parseInt(fileName.replace(MODEL_FILE_SUFFIX, ""));
    }

    public static void boardcast(String msg) throws IOException {
        DatagramSocket socket = new DatagramSocket();
        socket.setBroadcast(true);
        DatagramPacket packet = new DatagramPacket(msg.getBytes(), msg.getBytes().length,
                InetAddress.getByName("255.255.255.255"), BROADCAST_RECEIVER_PORT);
        socket.send(packet);
        socket.close();
    }
}
