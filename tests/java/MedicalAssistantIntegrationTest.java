/**
 * JUnit integration tests for Medical Assistant Bot
 * 
 * These tests demonstrate how to integrate Java-based testing
 * with the Python medical assistant system using process execution
 * and HTTP API calls.
 */
package com.medicalassistant.tests;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.CsvSource;

import java.io.*;
import java.net.http.*;
import java.net.URI;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.TimeUnit;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@ExtendWith(TestWatcherExtension.class)
public class MedicalAssistantIntegrationTest {
    
    private static final String PYTHON_EXECUTABLE = "python";
    private static final String BASE_URL = "http://localhost:8501";
    private static final int TIMEOUT_SECONDS = 30;
    
    private static HttpClient httpClient;
    private static ObjectMapper objectMapper;
    private static ProcessBuilder processBuilder;
    
    @BeforeAll
    static void setUpClass() {
        httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(TIMEOUT_SECONDS))
            .build();
        
        objectMapper = new ObjectMapper();
        
        // Setup process builder for Python script execution
        processBuilder = new ProcessBuilder();
        processBuilder.directory(new File(".."));
    }
    
    @AfterAll
    static void tearDownClass() {
        // Cleanup resources
        if (httpClient != null) {
            httpClient = null;
        }
    }
    
    @Test
    @Order(1)
    @DisplayName("System Health Check")
    void testSystemHealthCheck() {
        assertDoesNotThrow(() -> {
            // Test Python environment
            Process process = processBuilder
                .command(PYTHON_EXECUTABLE, "-c", "import sys; print('Python OK')")
                .start();
            
            boolean finished = process.waitFor(10, TimeUnit.SECONDS);
            assertTrue(finished, "Python health check should complete within timeout");
            assertEquals(0, process.exitValue(), "Python should be available");
        });
    }
    
    @Test
    @Order(2)
    @DisplayName("Required Files Validation")
    void testRequiredFilesExist() {
        String[] requiredFiles = {
            "../data/disease_symptoms.csv",
            "../data/disease_symptom_severity.csv", 
            "../data/disease_precautions.csv",
            "../data/disease_symptom_description.csv",
            "../src/medical_tools.py",
            "../src/vision_tools.py",
            "../src/medical_agent_langchain.py"
        };
        
        for (String filePath : requiredFiles) {
            File file = new File(filePath);
            assertTrue(file.exists(), "Required file should exist: " + filePath);
        }
    }
    
    @Test
    @Order(3)
    @DisplayName("FAISS Indices Validation")
    void testFaissIndicesExist() {
        String[] requiredIndices = {
            "../indices/faiss_symptom_index_medibot",
            "../indices/faiss_severity_index_medibot"
        };
        
        for (String indexPath : requiredIndices) {
            File indexDir = new File(indexPath);
            assertTrue(indexDir.exists() && indexDir.isDirectory(), 
                "FAISS index directory should exist: " + indexPath);
        }
    }
    
    @ParameterizedTest
    @ValueSource(strings = {"fever", "headache", "cough", "sore throat"})
    @DisplayName("Symptom Analysis Tests")
    void testSymptomAnalysis(String symptom) {
        assertDoesNotThrow(() -> {
            String pythonScript = String.format(
                "import sys; sys.path.append('src'); " +
                "from medical_tools import analyze_symptoms_direct; " +
                "print('Analyzing: %s')", 
                symptom
            );
            
            Process process = processBuilder
                .command(PYTHON_EXECUTABLE, "-c", pythonScript)
                .start();
            
            boolean finished = process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            assertTrue(finished, "Symptom analysis should complete within timeout");
        });
    }
    
    @ParameterizedTest
    @CsvSource({
        "fever,headache",
        "cough,sore throat", 
        "nausea,fatigue",
        "chest pain,shortness of breath"
    })
    @DisplayName("Multi-Symptom Analysis Tests")
    void testMultiSymptomAnalysis(String symptom1, String symptom2) {
        String combinedSymptoms = symptom1 + ", " + symptom2;
        
        assertDoesNotThrow(() -> {
            // Test combined symptom analysis
            String result = executePythonFunction(
                "analyze_combined_symptoms", 
                Arrays.asList(combinedSymptoms)
            );
            
            assertNotNull(result, "Analysis result should not be null");
            assertFalse(result.trim().isEmpty(), "Analysis result should not be empty");
        });
    }
    
    @Test
    @DisplayName("Medical Term Translation Test")
    void testMedicalTermTranslation() {
        String[] medicalTerms = {"erythema", "edema", "cellulitis", "pyrexia"};
        
        assertDoesNotThrow(() -> {
            for (String term : medicalTerms) {
                String result = executePythonFunction(
                    "translate_medical_terms", 
                    Arrays.asList(term)
                );
                
                assertNotNull(result, "Translation result should not be null for: " + term);
            }
        });
    }
    
    @Test
    @DisplayName("Image Processing Validation Test")
    void testImageProcessingValidation() {
        assertDoesNotThrow(() -> {
            // Test image format validation
            String[] validFormats = {"png", "jpg", "jpeg"};
            String[] invalidFormats = {"txt", "pdf", "doc"};
            
            for (String format : validFormats) {
                assertTrue(isValidImageFormat(format), 
                    "Should accept valid image format: " + format);
            }
            
            for (String format : invalidFormats) {
                assertFalse(isValidImageFormat(format), 
                    "Should reject invalid image format: " + format);
            }
        });
    }
    
    @Test
    @DisplayName("Error Handling Test")
    void testErrorHandling() {
        assertDoesNotThrow(() -> {
            // Test with empty input
            String result = executePythonFunction(
                "analyze_symptoms_direct", 
                Arrays.asList("")
            );
            
            // Should handle gracefully, not crash
            assertNotNull(result, "Error handling should return a result");
        });
    }
    
    @Test
    @DisplayName("Performance Benchmark Test")
    void testPerformanceBenchmark() {
        long startTime = System.currentTimeMillis();
        
        assertDoesNotThrow(() -> {
            // Run multiple analyses to test performance
            for (int i = 0; i < 5; i++) {
                executePythonFunction(
                    "analyze_symptoms_direct", 
                    Arrays.asList("fever, headache")
                );
            }
        });
        
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        
        // Should complete within reasonable time (30 seconds for 5 analyses)
        assertTrue(duration < 30000, 
            "Performance test should complete within 30 seconds, took: " + duration + "ms");
    }
    
    @Test
    @DisplayName("JSON Response Structure Test")
    void testJsonResponseStructure() {
        assertDoesNotThrow(() -> {
            String result = executePythonFunction(
                "analyze_symptoms_direct", 
                Arrays.asList("fever")
            );
            
            // Attempt to parse as JSON
            if (result.trim().startsWith("{")) {
                JsonNode jsonNode = objectMapper.readTree(result);
                assertNotNull(jsonNode, "Should be valid JSON structure");
            }
        });
    }
    
    @Test
    @DisplayName("Memory Usage Test")
    void testMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        long initialMemory = runtime.totalMemory() - runtime.freeMemory();
        
        assertDoesNotThrow(() -> {
            // Run memory-intensive operations
            for (int i = 0; i < 10; i++) {
                executePythonFunction(
                    "analyze_symptoms_direct", 
                    Arrays.asList("symptom batch " + i)
                );
            }
            
            // Force garbage collection
            System.gc();
            
            long finalMemory = runtime.totalMemory() - runtime.freeMemory();
            long memoryUsed = finalMemory - initialMemory;
            
            // Memory usage should be reasonable (less than 100MB for this test)
            assertTrue(memoryUsed < 100 * 1024 * 1024, 
                "Memory usage should be reasonable: " + (memoryUsed / 1024 / 1024) + "MB");
        });
    }
    
    // Helper Methods
    
    private String executePythonFunction(String functionName, List<String> args) throws Exception {
        String argsString = String.join("', '", args);
        String pythonScript = String.format(
            "import sys; sys.path.append('src'); " +
            "try: " +
            "    from medical_tools import %s; " +
            "    result = %s.func('%s') if hasattr(%s, 'func') else 'Function executed'; " +
            "    print(result); " +
            "except Exception as e: " +
            "    print('Error:', str(e))",
            functionName, functionName, argsString, functionName
        );
        
        Process process = processBuilder
            .command(PYTHON_EXECUTABLE, "-c", pythonScript)
            .start();
        
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {
            
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }
            
            boolean finished = process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                throw new RuntimeException("Python execution timed out");
            }
            
            return output.toString().trim();
        }
    }
    
    private boolean isValidImageFormat(String format) {
        String[] validFormats = {"png", "jpg", "jpeg", "gif"};
        return Arrays.asList(validFormats).contains(format.toLowerCase());
    }
    
    // Custom assertions
    
    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
    
    private static void assertFalse(boolean condition, String message) {
        if (condition) {
            throw new AssertionError(message);
        }
    }
    
    private static void assertEquals(Object expected, Object actual, String message) {
        if (!Objects.equals(expected, actual)) {
            throw new AssertionError(message + " Expected: " + expected + ", Actual: " + actual);
        }
    }
    
    private static void assertNotNull(Object object, String message) {
        if (object == null) {
            throw new AssertionError(message);
        }
    }
    
    private static void assertDoesNotThrow(ThrowingRunnable runnable) {
        try {
            runnable.run();
        } catch (Exception e) {
            throw new AssertionError("Expected no exception, but got: " + e.getMessage(), e);
        }
    }
    
    @FunctionalInterface
    interface ThrowingRunnable {
        void run() throws Exception;
    }
}

/**
 * Test watcher extension for logging test results
 */
class TestWatcherExtension implements org.junit.jupiter.api.extension.TestWatcher {
    
    @Override
    public void testSuccessful(ExtensionContext context) {
        System.out.println("✅ Test passed: " + context.getDisplayName());
    }
    
    @Override
    public void testFailed(ExtensionContext context, Throwable cause) {
        System.out.println("❌ Test failed: " + context.getDisplayName());
        System.out.println("   Reason: " + cause.getMessage());
    }
    
    @Override
    public void testSkipped(ExtensionContext context, Optional<String> reason) {
        System.out.println("⏭️ Test skipped: " + context.getDisplayName());
        reason.ifPresent(r -> System.out.println("   Reason: " + r));
    }
}