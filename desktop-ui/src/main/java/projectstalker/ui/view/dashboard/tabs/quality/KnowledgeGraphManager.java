package projectstalker.ui.view.dashboard.tabs.quality;

import javafx.application.Platform;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import lombok.extern.slf4j.Slf4j;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Record;
import org.neo4j.driver.Session;
import org.neo4j.driver.types.Node;
import org.neo4j.driver.types.Relationship;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

@Slf4j
public class KnowledgeGraphManager {

    private final StackPane container;
    private Canvas canvas;
    private Driver driver;

    // Graph Data
    private final Map<Long, GraphNode> nodes = new HashMap<>();
    private final List<GraphEdge> edges = new ArrayList<>();

    private boolean isRunning = false;
    private final Random random = new Random();

    public KnowledgeGraphManager(StackPane container) {
        this.container = container;
        initCanvas();
    }

    private void initCanvas() {
        this.canvas = new Canvas();
        this.canvas.widthProperty().bind(container.widthProperty());
        this.canvas.heightProperty().bind(container.heightProperty());
        container.getChildren().add(canvas);

        this.canvas.widthProperty().addListener(obs -> requestRender());
        this.canvas.heightProperty().addListener(obs -> requestRender());
    }

    public void connect(String uri, String user, String password) {
        try {
            this.driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password));
            log.info("Neo4j Connected: {}", uri);
            fetchGraph();
        } catch (Exception e) {
            log.error("Failed to connect to Neo4j", e);
            renderError("Connection Failed: " + e.getMessage());
        }
    }

    public void fetchGraph() {
        new Thread(() -> {
            try (Session session = driver.session()) {
                String query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 300";
                List<Record> records = session.run(query).list();

                log.info("Fetched {} records from Neo4j", records.size());

                synchronized (this) {
                    nodes.clear();
                    edges.clear();

                    for (Record rec : records) {
                        Node n = rec.get("n").asNode();
                        Node m = rec.get("m").asNode();
                        Relationship r = rec.get("r").asRelationship();

                        addNode(n);
                        addNode(m);

                        edges.add(new GraphEdge(n.id(), m.id(), r.type()));
                    }
                }

                // Init Random Positions
                double width = Math.max(800, canvas.getWidth());
                double height = Math.max(600, canvas.getHeight());

                nodes.values().forEach(node -> {
                    node.x = random.nextDouble() * width;
                    node.y = random.nextDouble() * height;
                });

                startSimulation();

            } catch (Exception e) {
                log.error("Error fetching graph", e);
                Platform.runLater(() -> renderError("Fetch Error"));
            }
        }).start();
    }

    private void addNode(Node n) {
        if (!nodes.containsKey(n.id())) {
            String label = "Node";
            if (n.labels().iterator().hasNext())
                label = n.labels().iterator().next();

            String name = String.valueOf(n.id());
            if (n.containsKey("name"))
                name = n.get("name").asString();
            else if (n.containsKey("title"))
                name = n.get("title").asString();
            else if (n.containsKey("id"))
                name = n.get("id").asString();

            nodes.put(n.id(), new GraphNode(n.id(), label, name));
        }
    }

    private void startSimulation() {
        isRunning = true;
        new Thread(() -> {
            while (isRunning) {
                updatePhysics();
                Platform.runLater(this::render);
                try {
                    Thread.sleep(30); // ~30 FPS
                } catch (InterruptedException e) {
                    break;
                }
            }
        }).start();
    }

    public void stop() {
        isRunning = false;
        if (driver != null)
            driver.close();
    }

    private void updatePhysics() {
        double width = canvas.getWidth();
        double height = canvas.getHeight();
        if (width == 0)
            width = 800;
        if (height == 0)
            height = 600;

        // Simple Force Directed Layout
        double repulsion = 5000;
        double attraction = 0.05;

        synchronized (this) {
            // Repulsion
            for (GraphNode n1 : nodes.values()) {
                n1.dx = 0;
                n1.dy = 0;
                for (GraphNode n2 : nodes.values()) {
                    if (n1 == n2)
                        continue;
                    double dx = n1.x - n2.x;
                    double dy = n1.y - n2.y;
                    double distSq = dx * dx + dy * dy;
                    if (distSq < 1)
                        distSq = 1;

                    double force = repulsion / distSq;
                    n1.dx += (dx / Math.sqrt(distSq)) * force;
                    n1.dy += (dy / Math.sqrt(distSq)) * force;
                }
            }

            // Attraction
            for (GraphEdge edge : edges) {
                GraphNode source = nodes.get(sourceId(edge));
                GraphNode target = nodes.get(targetId(edge));
                if (source == null || target == null)
                    continue;

                double dx = target.x - source.x;
                double dy = target.y - source.y;

                source.dx += dx * attraction;
                source.dy += dy * attraction;

                target.dx -= dx * attraction;
                target.dy -= dy * attraction;
            }

            // Apply & Boundary
            for (GraphNode n : nodes.values()) {
                n.x += n.dx * 0.1;
                n.y += n.dy * 0.1;

                // Keep inside (soft)
                if (n.x < 50)
                    n.dx += 5;
                if (n.x > width - 50)
                    n.dx -= 5;
                if (n.y < 50)
                    n.dy += 5;
                if (n.y > height - 50)
                    n.dy -= 5;
            }
        }
    }

    // Helper for Edge
    private long sourceId(GraphEdge e) {
        return e.source;
    }

    private long targetId(GraphEdge e) {
        return e.target;
    }

    private void render() {
        if (canvas == null)
            return;
        GraphicsContext gc = canvas.getGraphicsContext2D();
        double w = canvas.getWidth();
        double h = canvas.getHeight();

        gc.setFill(Color.web("#111"));
        gc.fillRect(0, 0, w, h);

        synchronized (this) {
            // Edges
            gc.setStroke(Color.web("#555", 0.6));
            gc.setLineWidth(1);
            for (GraphEdge edge : edges) {
                GraphNode src = nodes.get(edge.source);
                GraphNode tgt = nodes.get(edge.target);
                if (src != null && tgt != null) {
                    gc.strokeLine(src.x, src.y, tgt.x, tgt.y);
                }
            }

            // Nodes
            for (GraphNode node : nodes.values()) {
                gc.setFill(getColorForLabel(node.label));
                gc.fillOval(node.x - 10, node.y - 10, 20, 20);

                gc.setFill(Color.WHITE);
                gc.fillText(node.name, node.x + 12, node.y + 5);
            }
        }
    }

    private void requestRender() {
        render();
    }

    private void renderError(String msg) {
        if (canvas == null)
            return;
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setFill(Color.RED);
        gc.fillText(msg, 50, 50);
    }

    private Color getColorForLabel(String label) {
        // Simple hash color
        int hash = label.hashCode();
        return Color.hsb((hash & 0xFF), 0.7, 0.9);
    }

    // --- Inner Classes ---
    private static class GraphNode {
        long id;
        String label;
        String name;
        double x, y, dx, dy;

        public GraphNode(long id, String label, String name) {
            this.id = id;
            this.label = label;
            this.name = name;
        }
    }

    private static class GraphEdge {
        long source;
        long target;
        String type;

        public GraphEdge(long source, long target, String type) {
            this.source = source;
            this.target = target;
            this.type = type;
        }
    }
}
