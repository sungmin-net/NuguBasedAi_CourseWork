import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Scanner;


public class SyncNodeMain {

    private static final String NODE_NAME = "Node1";
    private static final String MODEL_FILE_SUFFIX = ".mdl";
    private static final String BROADCAST_ADDRESS = "192.168.0.255";
    private static final int BROADCAST_PORT = 50000;
    private static final int BRROADCAST_INTERVAL_MSEC = 1000;
    private static final int BUF_SIZE = 1024;
    private static final String TIME_STAMP_FORMAT = "yyMMddHHmmss";

    static String mCurModelFile = "0.mdl"; // default
    private static boolean mIsBroadcasting = false;
    private static boolean mIsListening = false;
    private static boolean mIsUpdating = false;

    static DatagramSocket mBroadcastSocket = null;

    static Thread mBroadcaster = new Thread(new Runnable() {

        @Override
        public void run() {
            System.out.println("# Broadcaster started.");
            mIsBroadcasting = true;
            try {
                while (mIsBroadcasting) {
                    // Pause broadcasting while updating
                    if (mIsUpdating) {
                        Thread.sleep(BRROADCAST_INTERVAL_MSEC);
                        continue;
                    }
                    refreshCurrentModelFile();
                    String msg = NODE_NAME + "|" + getTimeStamp() + "|" + mCurModelFile;
                    DatagramPacket packet = new DatagramPacket(msg.getBytes(),
                            msg.getBytes().length, InetAddress.getByName(BROADCAST_ADDRESS),
                            BROADCAST_PORT);
                    if (mBroadcastSocket.isClosed()) {
                        break;
                    }
                    mBroadcastSocket.send(packet);
                    System.out.println("# SENT: " + msg);
                    Thread.sleep(BRROADCAST_INTERVAL_MSEC);
                }
            } catch (InterruptedException | IOException e) {
                e.printStackTrace();
            } finally {
                System.out.println("# Broadcaster stopped.");
            }
        }
    });

    static Thread mListener = new Thread(new Runnable() {
        @Override
        public void run() {
            System.out.println("# Listener started.");
            mIsListening = true;
            try {
                while (mIsListening) {

                    byte[] buf = new byte[ BUF_SIZE ];
                    DatagramPacket packet = new DatagramPacket(buf, buf.length);
                    if (mBroadcastSocket.isClosed()) {
                        break;
                    }

                    mBroadcastSocket.receive(packet);
                    String msg = new String(packet.getData(), 0, packet.getLength());

                    // skip my msg
                    if (msg.startsWith(NODE_NAME)) {
                        continue;
                    }

                    // TODO Discard old msg

                    // TODO Discard old model

                    System.out.println("# RECEIVED: " + msg);

                    // parse peer msg
                    String[] msgParsing = msg.split("\\|");
                    String peerName = msgParsing[0];
                    String peerTimeStamp = msgParsing[1];
                    String peerModel = null;
                    String peerServIp = null;
                    int peerServPort = -1;
                    String peerSign = null; // TODO
                    int readBytes = -1;

                    // TODO verify signature

                    if (msgParsing[2].endsWith(MODEL_FILE_SUFFIX)) {
                        peerModel = msgParsing[2];
                        long peerVer = getVersion(peerModel);
                        if (peerVer > getVersion(mCurModelFile)) {
                            // CASE#1 - Peer's model is newer than mine
                            System.out.println("# Model update started.");
                            mIsUpdating = true;

                            // prepare model file receiver
                            ServerSocket servSock = new ServerSocket(0);

                            // send model file request msg
                            // TODO send RSA encrypted AES key together
                            InetAddress peerAddr = packet.getAddress();
                            String reqMsg = NODE_NAME + "|" + getTimeStamp() + "|" +
                                    InetAddress.getLocalHost().getHostAddress() + "|" +
                                    servSock.getLocalPort();
                            DatagramPacket reqPacket = new DatagramPacket(reqMsg.getBytes(),
                                    reqMsg.getBytes().length, peerAddr, BROADCAST_PORT);
                            mBroadcastSocket.send(reqPacket);
                            System.out.println("# Requested " + peerName + " to send its model.");

                            // receive peer's model
                            Socket recvSock = servSock.accept();
                            FileOutputStream fos = new FileOutputStream(peerModel);
                            InputStream is = recvSock.getInputStream();

                            while ((readBytes = is.read(buf)) != -1) {
                                fos.write(buf, 0, readBytes);
                            }
                            fos.close();
                            is.close();
                            recvSock.close();
                            mIsUpdating = false;
                        }
                    } else {
                        // CASE#2 - response peer's model request
                        peerServIp = msgParsing[2];
                        peerServPort = Integer.parseInt(msgParsing[3]);
                        // TODO send AES encrypted model
                        if (peerServIp != null && peerServPort != -1) {
                            FileInputStream fis = new FileInputStream(mCurModelFile);
                            Socket socket = new Socket(peerServIp, peerServPort);
                            OutputStream os = socket.getOutputStream();
                            while ((readBytes = fis.read(buf)) > 0) {
                                os.write(buf, 0, readBytes);
                            }
                            os.close();
                            fis.close();
                        }
                        System.out.println("# Sent " + mCurModelFile + " to " + peerName);
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                System.out.println("# Listener stopped.");
            }
        }
    });

    static long getVersion(String modelName) {
        return Long.parseLong(modelName.replace(MODEL_FILE_SUFFIX, ""));
    }

    public static void main(String[] args) {
//        System.out.println(getBroadcastMsg());

        try {
            mBroadcastSocket = new DatagramSocket(BROADCAST_PORT);
            mBroadcastSocket.setBroadcast(true);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
        mListener.start();
        mBroadcaster.start();

        System.out.println("Input \'q\' to quit:");
        Scanner scanner = new Scanner(System.in);
        String cmd = scanner.nextLine();
        if (cmd.equals("q")) {
            mIsBroadcasting = false;
            mIsListening = false;
            if (mBroadcastSocket != null && !mBroadcastSocket.isClosed()) {
                mBroadcastSocket.close();
            }
        }
    }

    static void refreshCurrentModelFile() {
        File curDir = new File(System.getProperty("user.dir"));
        long curVer = Long.parseLong(mCurModelFile.replace(MODEL_FILE_SUFFIX, ""));
        long maxVer = -1l;
        for (File file : curDir.listFiles()) {
            String fileName = file.getName();
            if (fileName.endsWith(MODEL_FILE_SUFFIX)) {
                maxVer = Math.max(maxVer, Long.parseLong(fileName.replace(MODEL_FILE_SUFFIX, "")));
            }
        }
        if (curVer < maxVer) {
            System.out.println("# UPDATED: from \"" + curVer + "\" to \"" + maxVer + "\"");
            mCurModelFile = maxVer + MODEL_FILE_SUFFIX;
        }
    }

    static String getTimeStamp() {
        SimpleDateFormat format = new SimpleDateFormat(TIME_STAMP_FORMAT);
        return format.format(new Date());
    }
}
