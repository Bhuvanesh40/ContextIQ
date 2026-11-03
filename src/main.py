
# ContextIQ - Main Application
# This entry point prints a short welcome and the project's purpose.

def main():
    # Decorative separators make the application name stand out in the terminal.
    print("=" * 45)
    print("          WELCOME TO CONTEXTIQ")
    print("=" * 45)

    # Describe the product and the broad goal of the document-search system.
    print("\nYour AI Knowledge Discovery Assistant")
    print("Project initialization successful!")

    print("\nProject Objective:")
    print("Organize, search, connect and explore")
    print("knowledge from multiple documents.")

    print("\nNext Step: Building the RAG pipeline.")


if __name__ == "__main__":
    # Run the greeting only when this file is executed directly, not imported.
    main()
