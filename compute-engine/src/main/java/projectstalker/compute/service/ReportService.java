package projectstalker.compute.service;

import com.lowagie.text.Document;
import com.lowagie.text.Paragraph;
import com.lowagie.text.pdf.PdfWriter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.Map;
import java.util.UUID;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

@Service
@Slf4j
@RequiredArgsConstructor
public class ReportService {

    // Simple in-memory job store
    private final Map<String, Map<String, Object>> jobs = new ConcurrentHashMap<>();

    private final projectstalker.compute.repository.ReportRepository reportRepository;
    private final projectstalker.compute.repository.AlertRepository alertRepository;

    @org.springframework.transaction.annotation.Transactional
    public projectstalker.compute.entity.ReportEntity createReport(
            projectstalker.domain.dto.report.CreateReportRequest request) {
        // 1. Create Report
        var report = projectstalker.compute.entity.ReportEntity.builder()
                .title(request.title())
                .body(request.body())
                .build();
        report = reportRepository.save(report);

        // 2. Link Alerts
        List<String> ids = request.alertIds();
        if (ids != null && !ids.isEmpty()) {
            List<projectstalker.compute.entity.AlertEntity> alerts = alertRepository.findAllById(ids);
            for (var alert : alerts) {
                alert.setStatus(projectstalker.compute.entity.AlertEntity.AlertStatus.RESOLVED);
                alert.setReport(report);
            }
            alertRepository.saveAll(alerts);
        }

        return report;
    }

    public String queueReportGeneration(Map<String, Object> criteria) {
        String jobId = UUID.randomUUID().toString();
        jobs.put(jobId, new ConcurrentHashMap<>(Map.of("status", "QUEUED")));

        // Trigger async processing
        generatePdfAsync(jobId, criteria);

        return jobId;
    }

    @Async
    public void generatePdfAsync(String jobId, Map<String, Object> criteria) {
        try {
            updateStatus(jobId, "PROCESSING");
            log.info("Starting PDF generation for Job {}", jobId);

            // Artificial delay to simulate heavy work
            Thread.sleep(2000);

            // Generate PDF using OpenPDF
            Document document = new Document();
            String fileName = "report_" + jobId + ".pdf";
            File file = new File(System.getProperty("java.io.tmpdir"), fileName);
            PdfWriter.getInstance(document, new FileOutputStream(file));

            document.open();
            document.add(new Paragraph("Reporte de Vertidos"));
            document.add(new Paragraph("Job ID: " + jobId));
            document.add(new Paragraph("Criterios: " + criteria.toString()));
            document.add(new Paragraph("Generado el: " + java.time.LocalDateTime.now()));
            document.close();

            // Update job with result
            Map<String, Object> job = jobs.get(jobId);
            job.put("status", "COMPLETED");
            job.put("downloadUrl", "/api/reports/download/" + fileName); // Assuming a download endpoint exists (not
                                                                         // implemented here)
            // Or just return the path for now
            job.put("filePath", file.getAbsolutePath());

            log.info("PDF generation completed for Job {}", jobId);

        } catch (Exception e) {
            log.error("Error generating PDF", e);
            updateStatus(jobId, "FAILED");
            jobs.get(jobId).put("error", e.getMessage());
        }
    }

    private void updateStatus(String jobId, String status) {
        Map<String, Object> job = jobs.get(jobId);
        if (job != null) {
            job.put("status", status);
        }
    }

    public Map<String, Object> getJobStatus(String id) {
        return jobs.getOrDefault(id, Map.of("status", "NOT_FOUND"));
    }
}
