// This script processes a CSV file containing emails, classifies them as ham or spam,
// formats them, and writes them to separate JSON output files.
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"strings"
	"sync"
)

// Email represents the structured email data for JSON output.
type Email struct {
	Text string `json:"text"`
}

// findColIndex finds the index of a target column name in the header row.
func findColIndex(header []string, target string) (int, error) {
	for i, colName := range header {
		if strings.ToLower(colName) == strings.ToLower(target) {
			return i, nil
		}
	}
	return -1, fmt.Errorf("column '%s' not found in header", target)
}

// worker processes rows from taskChan, formats them, and sends them to hamChan or spamChan.
// Adapted for processed_data 2.csv format: expects labelIdx, subjectIdx, emailToIdx, emailFromIdx, messageIdx
func worker(id int, taskChan <-chan []string, hamChan chan<- Email, spamChan chan<- Email, wg *sync.WaitGroup, labelIdx, subjectIdx, emailToIdx, emailFromIdx, messageIdx int) {
	defer wg.Done()
	log.Printf("Worker %d started", id)
	for row := range taskChan {
		// Basic check: Ensure label index is valid before accessing
		if labelIdx < 0 || labelIdx >= len(row) {
			log.Printf("Worker %d: Invalid label index %d for row with %d columns, skipping", id, labelIdx, len(row))
			continue
		}
		label := strings.ToLower(row[labelIdx])

		// Ensure all required data indices are valid
		indices := []int{subjectIdx, emailToIdx, emailFromIdx, messageIdx}
		fieldNames := []string{"subject", "email_to", "email_from", "message"}
		validIndices := true
		for i, idx := range indices {
			if idx < 0 || idx >= len(row) {
				log.Printf("Worker %d: Invalid data index %d (%s) for row with %d columns, skipping", id, idx, fieldNames[i], len(row))
				validIndices = false
				break
			}
		}
		if !validIndices {
			continue
		}

		// Extract data using relevant indices
		subject := row[subjectIdx]
		emailTo := row[emailToIdx]
		emailFrom := row[emailFromIdx]
		message := row[messageIdx]

		// Combine fields into the desired format, including headers
		// Format: [CLS] email_from: <from> [SEP] email_to: <to> [SEP] subject: <subject> [SEP] message: <message>
		formattedText := fmt.Sprintf("[CLS] email_from: %s [SEP] email_to: %s [SEP] subject: %s [SEP] message: %s", emailFrom, emailTo, subject, message)

		email := Email{Text: formattedText}

		// Send to appropriate channel based on label (assuming 0=ham, 1=spam)
		if label == "0" { // Treat '0' as ham
			// Add safety check for channel send? Maybe not necessary with buffered channels unless under extreme load.
			hamChan <- email
		} else if label == "1" { // Treat '1' as spam
			spamChan <- email
		} else {
			log.Printf("Worker %d: Unknown label '%s', skipping row", id, label)
		}
	}
	log.Printf("Worker %d finished", id)
}

func main() {
	// Configuration flag for number of workers
	// Benchmark different values on an L4 with 40 GB VRAM (for maximum parallel I/O).
	defaultNumWorkers := runtime.NumCPU() * 2
	numWorkers := flag.Int("workers", defaultNumWorkers, "Number of worker goroutines")
	// Default input set to processed_data 2.csv
	inputFile := flag.String("input", "datasets/processed_data 2.csv", "Path to the input CSV file")
	// Output filenames set to TREC2007 directory
	hamFile := flag.String("ham", "datasets/TREC2007/ham.json", "Path to the output ham JSON file")
	spamFile := flag.String("spam", "datasets/TREC2007/spam.json", "Path to the output spam JSON file")
	bufferSize := flag.Int("buffer", 1000, "Buffer size for channels")
	flag.Parse()

	log.Printf("Starting email processing with %d workers...", *numWorkers)

	// Open input CSV file
	csvFile, err := os.Open(*inputFile)
	if err != nil {
		log.Fatalf("Failed to open input CSV file '%s': %v", *inputFile, err)
	}
	defer csvFile.Close()

	// Create CSV reader
	reader := csv.NewReader(csvFile)
	reader.LazyQuotes = true // Handle potentially inconsistent quoting

	// Read header row
	header, err := reader.Read()
	if err != nil {
		log.Fatalf("Failed to read CSV header: %v", err)
	}

	// Find column indices dynamically for processed_data 2.csv format
	labelIdx, err := findColIndex(header, "label")
	if err != nil {
		log.Fatalf("Header error: %v", err)
	}
	subjectIdx, err := findColIndex(header, "subject")
	if err != nil {
		log.Fatalf("Header error: %v", err)
	}
	emailToIdx, err := findColIndex(header, "email_to") // Added back
	if err != nil {
		log.Fatalf("Header error: %v", err)
	}
	emailFromIdx, err := findColIndex(header, "email_from") // Added back
	if err != nil {
		log.Fatalf("Header error: %v", err)
	}
	messageIdx, err := findColIndex(header, "message") // Using message instead of body
	if err != nil {
		log.Fatalf("Header error: %v", err)
	}

	log.Printf("Found column indices: label=%d, subject=%d, email_to=%d, email_from=%d, message=%d", labelIdx, subjectIdx, emailToIdx, emailFromIdx, messageIdx)

	// Create channels
	taskChan := make(chan []string, *bufferSize)
	hamChan := make(chan Email, *bufferSize)
	spamChan := make(chan Email, *bufferSize)

	// WaitGroup for workers
	var wg sync.WaitGroup

	// Launch worker pool
	log.Printf("Launching %d workers...", *numWorkers)
	for i := 0; i < *numWorkers; i++ {
		wg.Add(1)
		// Pass the indices relevant to processed_data 2.csv format to the worker
		go worker(i+1, taskChan, hamChan, spamChan, &wg, labelIdx, subjectIdx, emailToIdx, emailFromIdx, messageIdx)
	}

	// Goroutine to read CSV and send rows to taskChan
	go func() {
		log.Println("CSV reader goroutine started...")
		for {
			record, err := reader.Read()
			if err == io.EOF {
				log.Println("CSV reader reached EOF.")
				break // End of file
			}
			if err != nil {
				log.Printf("Error reading CSV row: %v", err)
				// Decide if we should continue or stop based on the error type
				if _, ok := err.(*csv.ParseError); ok {
					log.Println("CSV parse error, continuing with next row...")
					continue // Skip problematic row
				} else {
					log.Println("Unrecoverable CSV read error, stopping reader.")
					break // Stop for other errors
				}
			}
			// Send row to task channel
			taskChan <- record
		}
		close(taskChan)
		log.Println("CSV reader goroutine finished, taskChan closed.")
	}()

	// Goroutine to wait for workers and close output channels
	go func() {
		log.Println("Waiting for workers to finish...")
		wg.Wait()
		log.Println("All workers finished. Closing output channels.")
		close(hamChan)
		close(spamChan)
	}()

	// WaitGroup for writer goroutines
	var writerWg sync.WaitGroup

	// Open output files and start writer goroutines
	outputFiles := map[string]chan Email{
		*hamFile:  hamChan,
		*spamFile: spamChan,
	}

	for filename, ch := range outputFiles {
		writerWg.Add(1)
		go func(outputFilename string, dataChan <-chan Email) {
			defer writerWg.Done()
			log.Printf("Writer goroutine for %s started.", outputFilename)
			file, err := os.Create(outputFilename)
			if err != nil {
				log.Printf("Failed to create output file '%s': %v", outputFilename, err)
				// Drain the channel to prevent deadlocks if file creation fails
				for range dataChan {
				}
				return
			}
			defer file.Close()

			encoder := json.NewEncoder(file)
			count := 0
			for email := range dataChan {
				if err := encoder.Encode(email); err != nil {
					log.Printf("Error encoding JSON to %s: %v", outputFilename, err)
					// Continue processing other emails if possible
				}
				count++
			}
			log.Printf("Writer goroutine for %s finished. Wrote %d emails.", outputFilename, count)
		}(filename, ch)
	}

	// Wait for writers to finish
	log.Println("Waiting for writer goroutines to finish...")
	writerWg.Wait()

	log.Println("Email processing completed successfully.")
}
