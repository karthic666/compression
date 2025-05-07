#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_COLORS 16777216 // 2^24 possible RGB colors

typedef struct {
    unsigned char r, g, b;
} Color;

typedef struct HuffmanNode {
    int color_index;
    int freq;
    struct HuffmanNode *left, *right;
} HuffmanNode;

typedef struct {
    char *codes[MAX_COLORS];
} HuffmanCodeTable;

typedef struct {
    unsigned char header[54];
    int width, height;
    Color *pixels;
} BMPImage;

// ----- Min-Heap for building Huffman Tree -----
#define HEAP_MAX 16777216
HuffmanNode* heap[HEAP_MAX];
int heap_size = 0;

void heap_push(HuffmanNode *node) {
    heap[++heap_size] = node;
    for (int i = heap_size; i > 1 && heap[i]->freq < heap[i/2]->freq; i /= 2) {
        HuffmanNode *tmp = heap[i]; heap[i] = heap[i/2]; heap[i/2] = tmp;
    }
}

HuffmanNode* heap_pop() {
    HuffmanNode *top = heap[1];
    heap[1] = heap[heap_size--];
    for (int i = 1; 2*i <= heap_size;) {
        int smallest = i;
        if (2*i <= heap_size && heap[2*i]->freq < heap[smallest]->freq) smallest = 2*i;
        if (2*i+1 <= heap_size && heap[2*i+1]->freq < heap[smallest]->freq) smallest = 2*i+1;
        if (smallest == i) break;
        HuffmanNode *tmp = heap[i]; heap[i] = heap[smallest]; heap[smallest] = tmp;
        i = smallest;
    }
    return top;
}

char *make_code(const char *base, char bit) {
    size_t len = strlen(base);
    char *new_code = malloc(len + 2);  // +1 for bit, +1 for null terminator
    if (!new_code) {
        perror("malloc failed");
        exit(1);
    }
    strcpy(new_code, base);
    new_code[len] = bit;
    new_code[len + 1] = '\0';
    return new_code;
}

// ----- Encode tree into code table -----
void generate_codes(HuffmanNode *node, char *code, HuffmanCodeTable *table) {
    if(!node)
        return;
    if (!node->left && !node->right) {
        table->codes[node->color_index] = strdup(code);
        return;
    }
    char *left_code = make_code(code, '0');
    char *right_code = make_code(code, '1');
    generate_codes(node->left, left_code, table);
    generate_codes(node->right, right_code, table);
    free(left_code);
    free(right_code);
}

// ----- Read BMP -----
BMPImage read_bmp(const char *filename) {
    FILE *f = fopen(filename, "rb");
    BMPImage img;
    fread(img.header, sizeof(unsigned char), 54, f);
    img.width = *(int*)&img.header[18];
    img.height = *(int*)&img.header[22];
    int size = img.width * img.height;
    img.pixels = malloc(size * sizeof(Color));
    for (int i = 0; i < size; i++) {
        fread(&img.pixels[i], 3, 1, f);
    }
    fclose(f);
    return img;
}

// ----- Write BMP -----
void write_bmp(const char *filename, BMPImage img) {
    FILE *f = fopen(filename, "wb");
    fwrite(img.header, sizeof(unsigned char), 54, f);
    fwrite(img.pixels, 3, img.width * img.height, f);
    fclose(f);
}

// ----- Main Compression -----
void encode_bmp(const char *infile, const char *outfile) {
    BMPImage img = read_bmp(infile);
    int *freq = calloc(MAX_COLORS, sizeof(int));
    int total = img.width * img.height;

    // Frequency count
    for (int i = 0; i < total; i++) {
        int idx = (img.pixels[i].r << 16) | (img.pixels[i].g << 8) | img.pixels[i].b;
        freq[idx]++;
    }

    // Build Huffman Tree
    for (int i = 0; i < MAX_COLORS; i++) {
       if (freq[i]) {
           HuffmanNode *node = malloc(sizeof(HuffmanNode));
           node->color_index = i;
           node->freq = freq[i];
           node->left = node->right = NULL;
           heap_push(node);
       }
    }

    while (heap_size > 2) {
       HuffmanNode *a = heap_pop();
       HuffmanNode *b = heap_pop();
       HuffmanNode *parent = malloc(sizeof(HuffmanNode));
       parent->color_index = -1;
       parent->freq = a->freq + b->freq;
       parent->left = a;
       parent->right = b;
       heap_push(parent);
    }

    HuffmanNode *root = heap_pop();
    HuffmanCodeTable table = {0};
    generate_codes(root, "", &table);

    // Encode and write to txt
    // FILE *out = fopen(outfile, "w");
    // fprintf(out, "%d %d\n", img.width, img.height); // header
    // for (int i = 0; i < total; i++) {
    //    int idx = (img.pixels[i].r << 16) | (img.pixels[i].g << 8) | img.pixels[i].b;
    //    fprintf(out, "%s", table.codes[idx]);
    // }
    // fclose(out);
    printf("Encoded image written to %s\n", outfile);
}

// ----- Decoder -----
void decode_bmp(const char *txtfile, const char *outfile, HuffmanNode *tree) {
    FILE *f = fopen(txtfile, "r");
    int w, h;
    fscanf(f, "%d %d\n", &w, &h);
    BMPImage img;
    memcpy(img.header, (unsigned char[54]){0x42,0x4D}, 54); // simple BMP header setup (should be properly initialized)
    *(int*)&img.header[18] = w;
    *(int*)&img.header[22] = h;
    img.width = w;
    img.height = h;
    img.pixels = malloc(w * h * sizeof(Color));

    HuffmanNode *node = tree;
    int ch, idx = 0;
    while ((ch = fgetc(f)) != EOF && idx < w * h) {
        node = (ch == '0') ? node->left : node->right;
        if (!node->left && !node->right) {
            Color c = { node->color_index >> 16 & 0xFF,
                        node->color_index >> 8 & 0xFF,
                        node->color_index & 0xFF };
            img.pixels[idx++] = c;
            node = tree;
        }
    }
    fclose(f);
    write_bmp(outfile, img);
    printf("Decoded image written to %s\n", outfile);
}

// ----- Entry Point -----
int main() {
    printf("encode bmp\n");
    encode_bmp("/home/naughtius-maximus/Desktop/ID/dp.bmp", "encoded.txt");

    // Rebuild tree from original image for decode (simplification — reuse encode for both)
    BMPImage tmp = read_bmp("input.bmp");
    int *freq = calloc(MAX_COLORS, sizeof(int));
    for (int i = 0; i < tmp.width * tmp.height; i++) {
        int idx = (tmp.pixels[i].r << 16) | (tmp.pixels[i].g << 8) | tmp.pixels[i].b;
        freq[idx]++;
    }
    for (int i = 0; i < MAX_COLORS; i++) {
        if (freq[i]) {
            HuffmanNode *node = malloc(sizeof(HuffmanNode));
            node->color_index = i;
            node->freq = freq[i];
            node->left = node->right = NULL;
            heap_push(node);
        }
    }
    while (heap_size > 1) {
        HuffmanNode *a = heap_pop();
        HuffmanNode *b = heap_pop();
        HuffmanNode *parent = malloc(sizeof(HuffmanNode));
        parent->color_index = -1;
        parent->freq = a->freq + b->freq;
        parent->left = a;
        parent->right = b;
        heap_push(parent);
    }
    HuffmanNode *tree_root = heap_pop();

    decode_bmp("encoded.txt", "decoded.bmp", tree_root);
    return 0;
}

