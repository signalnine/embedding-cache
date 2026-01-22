// Package vectorembed provides a client for the vector-embed-cache API.
//
// The Client is safe for concurrent use from multiple goroutines.
//
// Basic usage:
//
//	client, err := vectorembed.NewClient("your-api-key")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer client.Close()
//
//	embedding, err := client.Embed(ctx, "hello world")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Println(embedding.Vector)
package vectorembed

// Version is the client library version.
const Version = "0.1.0"
