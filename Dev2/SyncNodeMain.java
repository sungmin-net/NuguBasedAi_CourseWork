import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.security.InvalidKeyException;
import java.security.KeyManagementException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.PrivateKey;
import java.security.Signature;
import java.security.SignatureException;
import java.security.UnrecoverableKeyException;
import java.security.cert.Certificate;
import java.security.cert.CertificateEncodingException;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Base64;
import java.util.Date;
import java.util.Scanner;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLServerSocket;
import javax.net.ssl.SSLServerSocketFactory;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManagerFactory;


public class SyncNodeMain {

    private static final String NODE_NAME = "Dev2";
    private static final String MODEL_FILE_SUFFIX = ".mdl";
    private static final String BROADCAST_ADDRESS = "192.168.0.255"; // TODO adjust this if needed
    private static final int BROADCAST_PORT = 50000;
    private static final int BRROADCAST_INTERVAL_MSEC = 3000;
    private static final int BUF_SIZE = 2048;
    private static final SimpleDateFormat TIME_STAMP_FORMAT = new SimpleDateFormat("yyMMddHHmmss");
    private static final String KEYSTORE_PASS = "mmlabmmlab"; // TODO Protect this if needed later
    private static final int VALID_TIMESTAMP_THRESHOLD = 5000; // 5 sec.
    private static final String DELIMITER = "|";
    private static final String KEYSTORE_FILE = NODE_NAME + "Ks.p12";
    private static final String ROOT_ALIAS = "root";

    private static String mCurModelDir = "0.mdl"; // default
    private static boolean mIsBroadcasting = false;
    private static boolean mIsListening = false;
    private static boolean mIsUpdating = false;

    private static DatagramSocket mBroadcastSocket = null;
    private static KeyStore mKeyStore = null;
    private static SSLSocketFactory mSslSocketFactory = null;

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
                    refreshCurrentModel();
                    String msg = getBroadcastMsg();

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

        private String getBroadcastMsg() {
            StringBuffer buf = new StringBuffer();
            buf.append(NODE_NAME).append(DELIMITER)
            .append(getTimeStamp()).append(DELIMITER)
            .append(mCurModelDir);

            appendSignAndCert(buf);

            return buf.toString();
        }
    });

    static Thread mListener = new Thread(new Runnable() {
        @Override
        public void run() {
            System.out.println("# Listener started.");
            mIsListening = true;
            byte[] buf = new byte[ BUF_SIZE ];
            try {
                while (mIsListening) {
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

                    System.out.println("# RECEIVED: " + msg);

                    if (mIsUpdating) {
                        System.out.println("# DISCARDED: Node is being updated.");
                        continue;
                    }

                    // parse peer msg
                    String[] msgParsing = msg.split("\\" + DELIMITER);
                    String peerName = msgParsing[0];
                    String peerTimeStamp = msgParsing[1];

                    if (!isValidTimeStamp(peerTimeStamp)) {
                        System.out.println("# DISCARDED: " + peerName +
                                "\'s timestamp is timeout.");
                        continue;
                    }

                    byte[] peerCertBytes = Base64.getDecoder().decode(msgParsing[4] /*CERT*/);
                    CertificateFactory certFactory = CertificateFactory.getInstance("X.509");
                    Certificate peerCert = certFactory.generateCertificate(
                            new ByteArrayInputStream(peerCertBytes));
                    if (!isPeerCertVerified(peerCert)) {
                        System.out.println("# DISCARDED: " + peerName +
                                "\'s certificate cannot be verified.");
                        continue;
                    }

                    if (!isValidSignature(msgParsing, peerCert)) {
                        System.out.println(
                                "# DISCARDED: " + peerName + "\'s signature is not valid.");
                        continue;
                    }

                    int byteCounter = -1;
                    if (isBroadcastMsg(msgParsing[2])) {
                        // CASE#1 - handle peer's broadcast
                        String peerModel = msgParsing[2];

                        if (getVersion(peerModel) <= getVersion(mCurModelDir)) {
                            System.out.println("# DISCARDED: " + peerName + "\'s is not newer.");
                            continue;
                        }

                        System.out.println("# Model update started.");
                        mIsUpdating = true;

                        // prepare model file receiver via TLS
                        SSLServerSocketFactory sslServSockFactory =
                                (SSLServerSocketFactory) SSLServerSocketFactory.getDefault();
                        SSLServerSocket servSock =
                                (SSLServerSocket) sslServSockFactory.createServerSocket(0);

                        // send model file request msg
                        String reqMsg = getRequestMsg(InetAddress.getLocalHost().getHostAddress(),
                                servSock.getLocalPort());

                        InetAddress peerAddr = packet.getAddress();
                        DatagramPacket reqPacket = new DatagramPacket(reqMsg.getBytes(),
                                reqMsg.getBytes().length, peerAddr, BROADCAST_PORT);
                        mBroadcastSocket.send(reqPacket);
                        System.out.println("# Requested " + peerName + " to send its model.");

                        // receive peer's model
                        SSLSocket recvSock = (SSLSocket) servSock.accept();
                        FileOutputStream fos = new FileOutputStream(peerModel + ".zip");
                        InputStream is = recvSock.getInputStream();

                        while ((byteCounter = is.read(buf)) != -1) {
                            fos.write(buf, 0, byteCounter);
                        }
                        fos.close();
                        is.close();
                        recvSock.close();

                        // unzip it
                        ZipInputStream zis = new ZipInputStream(new FileInputStream(
                                peerModel + ".zip"));
                        ZipEntry zipEntry = zis.getNextEntry();
                        while (zipEntry != null) {
                            File newFile = unzipFile(new File(System.getProperty("user.dir")),
                                    zipEntry);
                            if (zipEntry.isDirectory()) {
                                if (!newFile.isDirectory() && !newFile.mkdirs()) {
                                    throw new IOException("Failed to create directory " + newFile);
                                }
                            } else {
                                // fix for Windows-created archives
                                File parent = newFile.getParentFile();
                                if (!parent.isDirectory() && !parent.mkdirs()) {
                                    throw new IOException("Failed to create directory " + parent);
                                }

                                // write file content
                                FileOutputStream os = new FileOutputStream(newFile);
                                int len;
                                while ((len = zis.read(buf)) > 0) {
                                    os.write(buf, 0, len);
                                }
                                os.close();
                            }
                            zipEntry = zis.getNextEntry();

                        }
                        zis.closeEntry();
                        zis.close();
                        new File(peerModel + ".zip").delete();

                        refreshCurrentModel();

                        mIsUpdating = false;

                    } else {
                        // CASE#2 - response peer's model request
                        String[] peerServInfo = msgParsing[2].split(":");
                        String peerServIp = peerServInfo[0];
                        int peerServPort = Integer.parseInt(peerServInfo[1]);

                        if (peerServIp != null && peerServPort != -1) {
                            String zipFileName = mCurModelDir + ".zip";
                            // zip current model directory
                            FileOutputStream fos = new FileOutputStream(zipFileName);
                            ZipOutputStream zipOut = new ZipOutputStream(fos);
                            File fileToZip = new File(mCurModelDir);

                            zipFile(fileToZip, fileToZip.getName(), zipOut);
                            zipOut.close();
                            fos.close();

                            // send it
                            FileInputStream fis = new FileInputStream(zipFileName);
                            SSLSocket socket = (SSLSocket) mSslSocketFactory.createSocket(
                                    peerServIp, peerServPort);
                            OutputStream os = socket.getOutputStream();
                            while ((byteCounter = fis.read(buf)) > 0) {
                                os.write(buf, 0, byteCounter);
                            }
                            os.close();
                            fis.close();

                            // remove it
                            new File(zipFileName).delete();

                        }
                        System.out.println("# Sent " + mCurModelDir + " to " + peerName);
                    }
                }
            } catch (IOException | CertificateException e) {
                e.printStackTrace();
            } finally {
                System.out.println("# Listener stopped.");
            }
        }

        private boolean isPeerCertVerified(Certificate peerCert) {

            try {
                Certificate ca = mKeyStore.getCertificate(ROOT_ALIAS);
                peerCert.verify(ca.getPublicKey());
                return true;
            } catch (KeyStoreException | InvalidKeyException | CertificateException |
                    NoSuchAlgorithmException | NoSuchProviderException | SignatureException e) {
                // DO nothing. just return false;
            }
            return false;
        }

        private String getRequestMsg(String localHost, int localPort) {
            StringBuffer buf = new StringBuffer();
            buf.append(NODE_NAME).append(DELIMITER)
                    .append(getTimeStamp()).append(DELIMITER)
                    .append(localHost).append(":")
                    .append(localPort);

            appendSignAndCert(buf);

            return buf.toString();
        }

        private long getVersion(String modelName) {
            return Long.parseLong(modelName.replace(MODEL_FILE_SUFFIX, ""));
        }

        private boolean isValidSignature(String[] msgParsing, Certificate peerCert) {

            try {
                StringBuffer buf = new StringBuffer();
                buf.append(msgParsing[0]/*NODE_NAME*/).append(DELIMITER)
                        .append(msgParsing[1]/*TIME_STAMP*/).append(DELIMITER)
                        .append(msgParsing[2]/*MODEL_NAME or SERV_INFO*/);
                byte[] signedBytes = buf.toString().getBytes("UTF8");
                byte[] signatureBytes = Base64.getDecoder().decode(msgParsing[3] /*SIGNATURE*/);

                /* TODO uses "RSASSA-PSS" if available
                Signature verifier = Signature.getInstance("SHA)
                verifier.setParameter(new PSSParameterSpec("SHA-256", "MGF1",
                        MGF1ParameterSpec.SHA256, 32, 1));
                */

                Signature verifier = Signature.getInstance("SHA256withRSA");
                verifier.initVerify(peerCert);
                verifier.update(signedBytes);
                return verifier.verify(signatureBytes);

            } catch (UnsupportedEncodingException | NoSuchAlgorithmException | InvalidKeyException
                    | SignatureException e) {
                e.printStackTrace();
            }

            return false;
        }

        private boolean isBroadcastMsg(String thirdField) {
            return thirdField.endsWith(MODEL_FILE_SUFFIX);
        }

        private boolean isValidTimeStamp(String peerTimeStamp) {
            try {
                Date target = TIME_STAMP_FORMAT.parse(peerTimeStamp);
                Date now = new Date();
                if (now.getTime() - target.getTime() < VALID_TIMESTAMP_THRESHOLD) {
                    return true;
                }
            } catch (ParseException e) {
                e.printStackTrace();
            }
            return false;
        }
    });

    static Certificate getDevCert() {
        Certificate ret = null;
        try {
            ret = mKeyStore.getCertificate(NODE_NAME);
        } catch (KeyStoreException e) {
            e.printStackTrace();
        }
        return ret;
    }

    static void appendSignAndCert(StringBuffer buf) {
        String signature = getSign(buf.toString());
        buf.append(DELIMITER).append(signature);

        Certificate cert = getDevCert();
        String certStr = null;
        try {
            certStr = Base64.getEncoder().encodeToString(cert.getEncoded());
        } catch (CertificateEncodingException e) {
            e.printStackTrace();
        }
        buf.append(DELIMITER).append(certStr);
    }

    private static String getSign(String toBeSigned) {
        String ret = null;
        try {
            PrivateKey privKey = (PrivateKey) mKeyStore.getKey(NODE_NAME,
                    KEYSTORE_PASS.toCharArray());

            /* TODO uses RSASSA-PSS if available
            Signature signer = Signature.getInstance("RSASSA-PSS");
            signer.setParameter(new PSSParameterSpec("SHA-256", "MGF1", MGF1ParameterSpec.SHA256,
                    32, 1));
            */

            Signature signer = Signature.getInstance("SHA256withRSA");
            signer.initSign(privKey);
            signer.update(toBeSigned.getBytes("UTF8"));
            byte[] signBytes = signer.sign();
            ret = Base64.getEncoder().encodeToString(signBytes);

        } catch (UnrecoverableKeyException | KeyStoreException | NoSuchAlgorithmException
                | InvalidKeyException| SignatureException | UnsupportedEncodingException e) {
            e.printStackTrace();
        }

        return ret;
    }

    private static void loadKeyStore() {
        try {
            mKeyStore = KeyStore.getInstance("PKCS12");
            FileInputStream fis = new FileInputStream(new File(KEYSTORE_FILE));
            mKeyStore.load(fis, KEYSTORE_PASS.toCharArray());

        } catch (KeyStoreException | IOException | NoSuchAlgorithmException |
                CertificateException e) {
            e.printStackTrace();
        }
    }

    private static void prepareSslSocketFactory() {
        try {
            // for SSLServerSocketFactory
            System.setProperty("javax.net.ssl.keyStore", KEYSTORE_FILE);
            System.setProperty("javax.net.ssl.keyStorePassword", KEYSTORE_PASS);
            // System.setProperty("javax.net.debug", "ssl");

            // for client socket
            TrustManagerFactory factory = TrustManagerFactory.getInstance(
                    TrustManagerFactory.getDefaultAlgorithm());
            factory.init(mKeyStore);
            SSLContext sslContext = SSLContext.getInstance("TLS");
            sslContext.init(null, factory.getTrustManagers(), null);
            mSslSocketFactory = sslContext.getSocketFactory();

        } catch (NoSuchAlgorithmException | KeyStoreException | KeyManagementException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {

        loadKeyStore();
        prepareSslSocketFactory();

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

    static void refreshCurrentModel() {
        File curDir = new File(System.getProperty("user.dir"));
        long curVer = Long.parseLong(mCurModelDir.replace(MODEL_FILE_SUFFIX, ""));
        long maxVer = -1l;
        for (File file : curDir.listFiles()) {
            String fileName = file.getName();
            if (fileName.endsWith(MODEL_FILE_SUFFIX)) {
                maxVer = Math.max(maxVer, Long.parseLong(fileName.replace(MODEL_FILE_SUFFIX, "")));
            }
        }
        if (curVer < maxVer) {
            System.out.println("# UPDATED: from \"" + curVer + "\" to \"" + maxVer + "\"");
            mCurModelDir = maxVer + MODEL_FILE_SUFFIX;
        }
    }

    static String getTimeStamp() {
        return TIME_STAMP_FORMAT.format(new Date()/*now*/);
    }

    static File unzipFile(File destinationDir, ZipEntry zipEntry) throws IOException {
        File destFile = new File(destinationDir, zipEntry.getName());

        String destDirPath = destinationDir.getCanonicalPath();
        String destFilePath = destFile.getCanonicalPath();

        if (!destFilePath.startsWith(destDirPath + File.separator)) {
            throw new IOException("Entry is outside of the target dir: " + zipEntry.getName());
        }

        return destFile;
    }

    static void zipFile(File fileToZip, String fileName, ZipOutputStream zipOut)
            throws IOException {
        if (fileToZip.isHidden()) {
            return;
        }
        if (fileToZip.isDirectory()) {
            if (fileName.endsWith("/")) {
                zipOut.putNextEntry(new ZipEntry(fileName));
                zipOut.closeEntry();
            } else {
                zipOut.putNextEntry(new ZipEntry(fileName + "/"));
                zipOut.closeEntry();
            }
            File[] children = fileToZip.listFiles();
            for (File childFile : children) {
                zipFile(childFile, fileName + "/" + childFile.getName(), zipOut);
            }
            return;
        }
        FileInputStream fis = new FileInputStream(fileToZip);
        ZipEntry zipEntry = new ZipEntry(fileName);
        zipOut.putNextEntry(zipEntry);
        byte[] bytes = new byte[1024];
        int length;
        while ((length = fis.read(bytes)) >= 0) {
            zipOut.write(bytes, 0, length);
        }
        fis.close();
    }
}
