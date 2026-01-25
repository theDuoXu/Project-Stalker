package projectstalker.ui.view.dashboard.tabs.quality;

import javafx.application.Platform;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
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
    private Label infoOverlay;

    // Graph Data
    private final Map<Long, GraphNode> nodes = new HashMap<>();
    private final List<GraphEdge> edges = new ArrayList<>();

    private boolean isRunning = false;
    private final Random random = new Random();

    // Interaction
    private GraphNode draggedNode = null;
    private GraphNode selectedNode = null;

    public KnowledgeGraphManager(StackPane container) {
        this.container = container;
        initUI();
    }

    private void initUI() {
        this.canvas = new Canvas();
        this.canvas.widthProperty().bind(container.widthProperty());
        this.canvas.heightProperty().bind(container.heightProperty());
        container.getChildren().add(canvas);

        // Overlay for Info
        this.infoOverlay = new Label("Click a node for details");
        this.infoOverlay.setStyle(
                "-fx-background-color: rgba(0,0,0,0.7); -fx-text-fill: white; -fx-padding: 10; -fx-background-radius: 5;");
        this.infoOverlay.setVisible(false);
        this.infoOverlay.setMouseTransparent(true); // Let clicks pass
        StackPane.setAlignment(infoOverlay, javafx.geometry.Pos.TOP_RIGHT);
        StackPane.setMargin(infoOverlay, new javafx.geometry.Insets(10));
        container.getChildren().add(infoOverlay);

        this.canvas.widthProperty().addListener(obs -> requestRender());
        this.canvas.heightProperty().addListener(obs -> requestRender());

        setupInteraction();
    }

    private void setupInteraction() {
        canvas.setOnMousePressed(e -> {
            GraphNode hit = findNodeAt(e.getX(), e.getY());
            if (hit != null) {
                draggedNode = hit;
                draggedNode.x = e.getX();
                draggedNode.y = e.getY();
                draggedNode.dx = 0;
                draggedNode.dy = 0;
                requestRender();
                // Wake up sim if sleeping
                if (!isRunning)
                    startSimulation();
            }
        });

        canvas.setOnMouseDragged(e -> {
            if (draggedNode != null) {
                draggedNode.x = e.getX();
                draggedNode.y = e.getY();
                draggedNode.dx = 0;
                draggedNode.dy = 0;
                requestRender();
            }
        });

        canvas.setOnMouseReleased(e -> {
            draggedNode = null;
        });

        canvas.setOnMouseClicked(e -> {
            if (e.isStillSincePress()) {
                GraphNode hit = findNodeAt(e.getX(), e.getY());
                selectedNode = hit;
                updateInfoOverlay();
                requestRender();
            }
        });
    }

    private GraphNode findNodeAt(double x, double y) {
        double radius = 15; // Hitbox
        synchronized (this) {
            for (GraphNode n : nodes.values()) {
                double dist = Math.sqrt(Math.pow(x - n.x, 2) + Math.pow(y - n.y, 2));
                if (dist <= radius) {
                    return n;
                }
            }
        }
        return null;
    }

    private void updateInfoOverlay() {
        if (selectedNode != null) {
            StringBuilder sb = new StringBuilder();
            sb.append("ID: ").append(selectedNode.id).append("\n");
            sb.append("Label: ").append(selectedNode.label).append("\n");
            sb.append("Name: ").append(selectedNode.name).append("\n");
            sb.append("-- Properties --\n");
            selectedNode.properties.forEach((k, v) -> {
                if (!"name".equals(k) && !"title".equals(k)) {
                    sb.append(k).append(": ").append(v).append("\n");
                }
            });
            infoOverlay.setText(sb.toString());
            infoOverlay.setVisible(true);
        } else {
            infoOverlay.setVisible(false);
        }
    }

    public void connect(String uri, String user, String password) {
        try {
            this.driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password));
            log.info("Neo4j Connected: {}", uri);
            fetchGraph("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 300");
        } catch (Exception e) {
            log.error("Failed to connect to Neo4j", e);
            renderError("Connection Failed: " + e.getMessage());
        }
    }

    public void fetchGraph(String customQuery) {
        new Thread(() -> {
            try (Session session = driver.session()) {
                String query = (customQuery == null || customQuery.isBlank())
                        ? "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 300"
                        : customQuery;

                List<Record> records = session.run(query).list();

                log.info("Fetched {} records from Neo4j", records.size());

                synchronized (this) {
                    nodes.clear();
                    edges.clear();

                    for (Record rec : records) {
                        for (String key : rec.keys()) {
                            org.neo4j.driver.Value value = rec.get(key);
                            String type = value.type().name();

                            if ("NODE".equals(type)) {
                                addNode(value.asNode());
                            } else if ("RELATIONSHIP".equals(type)) {
                                Relationship r = value.asRelationship();
                                edges.add(new GraphEdge(r.startNodeId(), r.endNodeId(), r.type()));
                            } else if ("PATH".equals(type)) {
                                org.neo4j.driver.types.Path p = value.asPath();
                                p.nodes().forEach(this::addNode);
                                p.relationships().forEach(
                                        r -> edges.add(new GraphEdge(r.startNodeId(), r.endNodeId(), r.type())));
                            }
                        }
                    }
                }

                // Init Random Positions
                double width = Math.max(800, canvas.getWidth());
                double height = Math.max(600, canvas.getHeight());

                synchronized (this) {
                    nodes.values().forEach(node -> {
                        node.x = random.nextDouble() * width;
                        node.y = random.nextDouble() * height;
                    });
                }

                startSimulation();

            } catch (Exception e) {
                log.error("Error fetching graph", e);
                Platform.runLater(() -> renderError("Fetch Error: " + e.getMessage()));
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

            nodes.put(n.id(), new GraphNode(n.id(), label, name, n.asMap()));
        }
    }

    private void startSimulation() {
        if (isRunning)
            return; // Already running
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
        double friction = 0.85; // Damping

        synchronized (this) {
            // Repulsion
            for (GraphNode n1 : nodes.values()) {
                // FIXED: Don't reset velocity if being dragged
                if (n1 == draggedNode)
                    continue;

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

                // Don't apply forces if dragged
                if (source != draggedNode) {
                    source.dx += dx * attraction;
                    source.dy += dy * attraction;
                }

                if (target != draggedNode) {
                    target.dx -= dx * attraction;
                    target.dy -= dy * attraction;
                }
            }

            // Apply & Boundary
            for (GraphNode n : nodes.values()) {
                if (n == draggedNode)
                    continue;

                // Apply Velocity
                n.x += n.dx * 0.1; // Simple integration
                n.y += n.dy * 0.1;

                // Friction
                n.dx *= friction;
                n.dy *= friction;

                // Keep inside (soft)
                if (n.x < 50)
                    n.x = 50; // Hard clamp for stability
                if (n.x > width - 50)
                    n.x = width - 50;
                if (n.y < 50)
                    n.y = 50;
                if (n.y > height - 50)
                    n.y = height - 50;
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
                // Highlight Selected
                if (node == selectedNode) {
                    gc.setStroke(Color.WHITE);
                    gc.setLineWidth(2);
                    gc.strokeOval(node.x - 12, node.y - 12, 24, 24);
                }

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
        Map<String, Object> properties;
        double x, y, dx, dy;

        public GraphNode(long id, String label, String name, Map<String, Object> properties) {
            this.id = id;
            this.label = label;
            this.name = name;
            this.properties = properties;
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
