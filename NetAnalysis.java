package cs1501_p4;

import java.util.ArrayList;
import java.util.Scanner;
import java.io.*;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class NetAnalysis implements NetAnalysis_Inter{
  //Constructor Variable
  private String file;
  //Class Constants
  private long COPPER = 230000000;
  private long FIBER = 200000000;
  //Class Variables
  private int verts;
  private node[] adjc;
  private EdgeWeightedDigraph G;
  private EdgeWeightedGraph Ug;
  /*
  *Constructs both an undirected and directed graph of the File
  *Uses an adjacency list representation array of linked lists.
  */
  public NetAnalysis(String file){
    this.file=file;

    try(Scanner infile = new Scanner(new File(file));){
      this.verts = infile.nextInt();
      adjc = new node[this.verts];
      G = new EdgeWeightedDigraph(this.verts);
      Ug = new EdgeWeightedGraph(this.verts);
      infile.nextLine();

      while(infile.hasNext()){
        String[] data = infile.nextLine().split(" ");
        int v=Integer.parseInt(data[0]);

        int w = Integer.parseInt(data[1]);
        double weight;
        String type = data[2];
        if(type.equals("copper")){
          weight = (double) Integer.parseInt(data[3]) / (double) COPPER;
        }
        else weight = (double) Integer.parseInt(data[3]) / (double) FIBER;
        DirectedEdge cur = new DirectedEdge(v,w,weight);
        Edge curr = new Edge(v,w,weight);

        G.addEdge(cur);
        Ug.addEdge(curr);
        add(Integer.parseInt(data[0]),Integer.parseInt(data[1]),data[2],Integer.parseInt(data[3]),Integer.parseInt(data[4]));
        add(Integer.parseInt(data[1]),Integer.parseInt(data[0]),data[2],Integer.parseInt(data[3]),Integer.parseInt(data[4]));
      }
    }
    catch(IOException e){
      e.printStackTrace();
    }
  }
  
  private void add(int n, int p, String c, int b, int l){
    if(adjc[n]==null){
      adjc[n] = new node(n);
      adjc[n].next = new node(p, c, b, l);
      return;
    }
    node curr = adjc[n];
    while(curr.next!=null){
      if(curr.point==p) return;
      curr=curr.next;
    }
    curr.next=new node(p,c,b,l);
  }

  /**
	 * Find the lowest latency path from vertex `u` to vertex `w` in the graph
	 *
	 * @param	u Starting vertex
	 * @param	w Destination vertex
	 *
	 * @return	ArrayList<Integer> A list of the vertex id's representing the
	 * 			path (should start with `u` and end with `w`)
	 * 			Return `null` if no path exists
	 */
	public ArrayList<Integer> lowestLatencyPath(int u, int w){
    //base case
    ArrayList<Integer> path = new ArrayList<Integer>();
    Iterable<DirectedEdge> start = G.adj(u);
    double[] dist = new double[G.V()];
    DirectedEdge[] pathTo = new DirectedEdge[G.V()];
    for(int i=0; i<dist.length; i++) dist[i]=Double.POSITIVE_INFINITY;
    dist[u]=0.0;
    //path.add(u);
    IndexMinPQ<Double> pq = new IndexMinPQ(G.V());
    pq.insert(u,0.0);
    while(!pq.isEmpty()){
      int v = pq.delMin();
      Iterable<DirectedEdge> st = G.adj(v);
      for(DirectedEdge e: st){
        int n = e.to();
        if(dist[n]>dist[v]+e.weight()){
          dist[n]=dist[v]+e.weight();
          pathTo[n]=e;
          if(pq.contains(n)) pq.changeKey(n,dist[n]);
          else pq.insert(n,dist[n]);
        }
      }
    }
    for(DirectedEdge e = pathTo[w]; e!=null; e=pathTo[e.from()]){
      path.add(0,e.from());
    }
    path.add(w);
    return path;
  }

	/**
	 * Find the bandwidth available along a given path through the graph
	 * (the minimum bandwidth of any edge in the path). Should throw an
	 * `IllegalArgumentException` if the specified path is not valid for
	 * the graph.
	 *
	 * @param	ArrayList<Integer> A list of the vertex id's representing the
	 * 			path
	 *
	 * @return	int The bandwidth available along the specified path
	 */
	public int bandwidthAlongPath(ArrayList<Integer> p) throws IllegalArgumentException{
    int s = p.get(0);
    node start = adjc[s];
    int currmin=Integer.MAX_VALUE;
    for(int i=1; i<p.size(); i++){
      node head=start;
      while(head!=null){
        head=head.next;
        if(head.point==p.get(i) && head.bandwith < currmin){
          currmin=head.bandwith;
          break;
        }
    }
    if(head==null) throw new IllegalArgumentException();
    start=adjc[p.get(i)];
  }

  return currmin;
}

	/**
	 * Return `true` if the graph is connected considering only copper links
	 * `false` otherwise
	 *
	 * @return	boolean Whether the graph is copper-only connected
	 */
	public boolean copperOnlyConnected(){
    boolean[] marks = new boolean[verts];
    coppersearch(0,marks);
    for(boolean m: marks){
      if(m==false) return false;
    }
    return true;
  }
  private void coppersearch(int n, boolean[] marks){
    node cur = adjc[n];
    marks[n]=true;
    for(node head=cur.next; head!=null; head=head.next){
      int piv = head.point;
      if(head.cable.equals("copper") && marks[piv]==false) coppersearch(piv,marks);
    }
  }

	/**
	 * Return `true` if the graph would remain connected if any two vertices in
	 * the graph would fail, `false` otherwise
	 *
	 * @return	boolean Whether the graph would remain connected for any two
	 * 			failed vertices
	 */
	public boolean connectedTwoVertFail(){

    boolean[] marks = new boolean[Ug.V()];
    boolean[] color = new boolean[Ug.V()];
    boolean bipartite=true;
    for(int s=0; s<Ug.V(); s++){
      if(!marks[s]) MDFS(s,marks,color,bipartite);
    }
    return bipartite;

  }
  private void MDFS(int s, boolean[] marks, boolean[] color,boolean bipartite){
    marks[s]=true;
    for(Edge w: Ug.adj(s)){
      if(!marks[w.either()]){
        color[w.either()]=!color[s];
        MDFS(w.either(),marks,color,bipartite);
      }
      else if(color[w.either()]==color[s]) bipartite=false;
    }
  }
  private void DFS(int n, boolean[] marks){
    node cur = adjc[n];
    marks[n]=true;
    for(node head=cur.next; head!=null; head=head.next){
      int piv = head.point;
      if(marks[piv]==false) DFS(piv,marks);
    }
  }

	/**
	 * Find the lowest average (mean) latency spanning tree for the graph
	 * (i.e., a spanning tree with the lowest average latency per edge). Return
	 * it as an ArrayList of STE edges.
	 *
	 * Note that you do not need to use the STE class to represent your graph
	 * internally, you only need to use it to construct return values for this
	 * method.
	 *
	 * @return	ArrayList<STE> A list of STE objects representing the lowest
	 * 			average latency spanning tree
	 * 			Return `null` if the graph is not connected
	 */
	public ArrayList<STE> lowestAvgLatST(){
    if(!isconnected()) return null;
    //initialize variables
    ArrayList<STE> MinSpanTree = new ArrayList<STE>();
    Edge[] pathTo = new Edge[Ug.V()];
    double[] dist = new double[Ug.V()];
    boolean[] marks = new boolean[Ug.V()];
    IndexMinPQ<Double> pq = new IndexMinPQ(Ug.V());
    //set all unknown weights to 0 by default
    for(int v=0; v<Ug.V(); v++) dist[v]=Double.POSITIVE_INFINITY;
    dist[0]=0.0;
    pq.insert(0,0.0);
    //use Prim's MST Algorithim (Eager Version) to calculate the MST
    while(!pq.isEmpty()){
      int v = pq.delMin();
      marks[v]=true;
      for(Edge e: Ug.adj(v)){
        int w = e.other(v);
        if(marks[w]) continue;
        if(e.weight()<dist[w]){
          pathTo[w]=e;
          dist[w]=e.weight();
          if(pq.contains(w)) pq.changeKey(w,dist[w]);
          else pq.insert(w,dist[w]);
        }
      }
    }
    //format for the output
    for(int i=1; i<pathTo.length; i++){
      Edge e = pathTo[i];
      STE cur = new STE(e.either(),e.other(e.either()));
      MinSpanTree.add(cur);
    }

    return MinSpanTree;
  }
  private boolean isconnected(){
    boolean[] marks = new boolean[verts];
    DFS(0,marks);
    for(boolean m: marks){
      if(m==false) return false;
    }
    return true;
  }

}
class node{
  int point;
  String cable;
  int bandwith;
  int len;
  node next;
  public node(int p){
    this.point=p;
    node next=null;
  }
  public node(int p, String c, int b, int n){
    this.point=p;
    this.cable=c;
    this.bandwith=b;
    this.len=n;
    this.next=null;
  }
}

/*
*ALL OF THE FOLLOWING IS PROVIDED BY  @author Robert Sedgewick @author Kevin Wayne
* PERMITTED USAGE AS PER PROJECT GUIDLINES ON GITHUB
*/
class EdgeWeightedDigraph{
  private final int V;
  private int E;
  private Bag<DirectedEdge>[] adj;
  public EdgeWeightedDigraph(int V){
    this.V=V;
    this.E=0;
    adj= (Bag<DirectedEdge>[]) new Bag[V];
    for(int v=0; v<V; v++) adj[v]= new Bag<DirectedEdge>();
  }
  public int V(){ return this.V;}
  public void addEdge(DirectedEdge e){
    adj[e.from()].add(e);
    E++;
  }
  public Iterable<DirectedEdge> adj(int v){
    return adj[v];
  }
  public Iterable<DirectedEdge> edges(){
    Bag<DirectedEdge> bag = new Bag<DirectedEdge>();
    for(int v=0; v<V; v++){
      for(DirectedEdge e: adj[v]){

        bag.add(e);
      }
      //bag.add(e);
    }
    return bag;
  }

}

class Bag<Item> implements Iterable<Item>{
  private Node first;
  private class Node{
    Item item;
    Node next;
  }
  public void add(Item item){
    Node oldfirst=first;
    first = new Node();
    first.item=item;
    first.next=oldfirst;
  }
  public Iterator<Item> iterator(){
    return new ListIterator();
  }
  private class ListIterator implements Iterator<Item>{
    private Node current=first;
    public boolean hasNext(){return current!=null;}
    public void remove(){

    }
    public Item next(){
      Item item = current.item;
      current = current.next;
      return item;
    }
  }
}
class DirectedEdge{
  private final int v;
  private final int w;
  private final double weight;
  public DirectedEdge(int v, int w, double weight){
    this.v=v;
    this.w=w;
    this.weight=weight;
  }
  public double weight(){return weight;}
  public int from(){return v;}
  public int to(){return w;}
  public String toString(){
    return String.format("%d->%d %.11f", v,w,weight);
  }

}

class EdgeWeightedGraph{
  private final int V;
  private int E;
  private Bag<Edge>[] adj;
  public EdgeWeightedGraph(int V){
    this.V=V;
    this.E=0;
    adj=(Bag<Edge>[]) new Bag[V];
    for(int i=0; i<V; i++) adj[i]=new Bag<Edge>();
  }
  public int V(){
    return this.V;
  }
  public int E(){
    return this.E;
  }
  public void addEdge(Edge e){
    int v = e.either();
    int w = e.other(v);
    adj[v].add(e);
    adj[w].add(e);
    E++;
  }
  public Iterable<Edge> adj(int v){
    return adj[v];
  }
  public Iterable<Edge> edges(){
    Bag<Edge> bag = new Bag<Edge>();
    for(int v=0; v<V; v++){
      for(Edge e: adj[v]){
        if(e.other(v)>v) bag.add(e);
      }

    }
    return bag;
  }
}
class Edge implements Comparable<Edge>{
  private final int v;
  private final int w;
  private final double weight;
  public Edge(int v, int w, double weight){
    this.v=v;
    this.w=w;
    this.weight=weight;

  }
  public double weight(){
    return this.weight;
  }
  public int either(){
    return this.v;
  }
  public int other(int vert){
    if(vert==v) return w;
    else if(vert==w) return v;
    else throw new IllegalArgumentException();
  }
  public int compareTo(Edge that){
    if(this.weight()<that.weight()) return -1;
    else if(this.weight()>that.weight()) return +1;
    else return 0;
  }

  public String toString(){
    return String.format("%d-%d %.11f", v, w, weight);
  }

}



/**
 *  @author Robert Sedgewick
 *  @author Kevin Wayne
 *  @param <Key> the generic type of key on this priority queue
 */
class IndexMinPQ<Key extends Comparable<Key>> {
    private int maxN;        // maximum number of elements on PQ
    private int n;           // number of elements on PQ
    private int[] pq;        // binary heap using 1-based indexing
    private int[] qp;        // inverse of pq - qp[pq[i]] = pq[qp[i]] = i
    private Key[] keys;      // keys[i] = priority of i

    /**
     * Initializes an empty indexed priority queue with indices between {@code 0}
     * and {@code maxN - 1}.
     * @param  maxN the keys on this priority queue are index from {@code 0}
     *         {@code maxN - 1}
     * @throws IllegalArgumentException if {@code maxN < 0}
     */
    public IndexMinPQ(int maxN) {
        if (maxN < 0) throw new IllegalArgumentException();
        this.maxN = maxN;
        n = 0;
        keys = (Key[]) new Comparable[maxN + 1];    // make this of length maxN??
        pq   = new int[maxN + 1];
        qp   = new int[maxN + 1];                   // make this of length maxN??
        for (int i = 0; i <= maxN; i++)
            qp[i] = -1;
    }

    /**
     * Returns true if this priority queue is empty.
     *
     * @return {@code true} if this priority queue is empty;
     *         {@code false} otherwise
     */
    public boolean isEmpty() {
        return n == 0;
    }

    /**
     * Is {@code i} an index on this priority queue?
     *
     * @param  i an index
     * @return {@code true} if {@code i} is an index on this priority queue;
     *         {@code false} otherwise
     * @throws IllegalArgumentException unless {@code 0 <= i < maxN}
     */
    public boolean contains(int i) {
        validateIndex(i);
        return qp[i] != -1;
    }

    /**
     * Returns the number of keys on this priority queue.
     *
     * @return the number of keys on this priority queue
     */
    public int size() {
        return n;
    }

    /**
     * Associates key with index {@code i}.
     *
     * @param  i an index
     * @param  key the key to associate with index {@code i}
     * @throws IllegalArgumentException unless {@code 0 <= i < maxN}
     * @throws IllegalArgumentException if there already is an item associated
     *         with index {@code i}
     */
    public void insert(int i, Key key) {
        validateIndex(i);
        if (contains(i)) throw new IllegalArgumentException("index is already in the priority queue");
        n++;
        qp[i] = n;
        pq[n] = i;
        keys[i] = key;
        swim(n);
    }

    /**
     * Returns an index associated with a minimum key.
     *
     * @return an index associated with a minimum key
     * @throws NoSuchElementException if this priority queue is empty
     */
    public int minIndex() {
        if (n == 0) throw new NoSuchElementException("Priority queue underflow");
        return pq[1];
    }

    /**
     * Returns a minimum key.
     *
     * @return a minimum key
     * @throws NoSuchElementException if this priority queue is empty
     */
    public Key minKey() {
        if (n == 0) throw new NoSuchElementException("Priority queue underflow");
        return keys[pq[1]];
    }

    /**
     * Removes a minimum key and returns its associated index.
     * @return an index associated with a minimum key
     * @throws NoSuchElementException if this priority queue is empty
     */
    public int delMin() {
        if (n == 0) throw new NoSuchElementException("Priority queue underflow");
        int min = pq[1];
        exch(1, n--);
        sink(1);
        assert min == pq[n+1];
        qp[min] = -1;        // delete
        keys[min] = null;    // to help with garbage collection
        pq[n+1] = -1;        // not needed
        return min;
    }

    /**
     * Returns the key associated with index {@code i}.
     *
     * @param  i the index of the key to return
     * @return the key associated with index {@code i}
     * @throws IllegalArgumentException unless {@code 0 <= i < maxN}
     * @throws NoSuchElementException no key is associated with index {@code i}
     */
    public Key keyOf(int i) {
        validateIndex(i);
        if (!contains(i)) throw new NoSuchElementException("index is not in the priority queue");
        else return keys[i];
    }

    /**
     * Change the key associated with index {@code i} to the specified value.
     *
     * @param  i the index of the key to change
     * @param  key change the key associated with index {@code i} to this key
     * @throws IllegalArgumentException unless {@code 0 <= i < maxN}
     * @throws NoSuchElementException no key is associated with index {@code i}
     */
    public void changeKey(int i, Key key) {
        validateIndex(i);
        if (!contains(i)) throw new NoSuchElementException("index is not in the priority queue");
        keys[i] = key;
        swim(qp[i]);
        sink(qp[i]);
    }

    /**
     * Change the key associated with index {@code i} to the specified value.
     *
     * @param  i the index of the key to change
     * @param  key change the key associated with index {@code i} to this key
     * @throws IllegalArgumentException unless {@code 0 <= i < maxN}
     * @deprecated Replaced by {@code changeKey(int, Key)}.
     */
    @Deprecated
    public void change(int i, Key key) {
        changeKey(i, key);
    }

    /**
     * Decrease the key associated with index {@code i} to the specified value.
     *
     * @param  i the index of the key to decrease
     * @param  key decrease the key associated with index {@code i} to this key
     * @throws IllegalArgumentException unless {@code 0 <= i < maxN}
     * @throws IllegalArgumentException if {@code key >= keyOf(i)}
     * @throws NoSuchElementException no key is associated with index {@code i}
     */
    public void decreaseKey(int i, Key key) {
        validateIndex(i);
        if (!contains(i)) throw new NoSuchElementException("index is not in the priority queue");
        if (keys[i].compareTo(key) == 0)
            throw new IllegalArgumentException("Calling decreaseKey() with a key equal to the key in the priority queue");
        if (keys[i].compareTo(key) < 0)
            throw new IllegalArgumentException("Calling decreaseKey() with a key strictly greater than the key in the priority queue");
        keys[i] = key;
        swim(qp[i]);
    }

    /**
     * Increase the key associated with index {@code i} to the specified value.
     *
     * @param  i the index of the key to increase
     * @param  key increase the key associated with index {@code i} to this key
     * @throws IllegalArgumentException unless {@code 0 <= i < maxN}
     * @throws IllegalArgumentException if {@code key <= keyOf(i)}
     * @throws NoSuchElementException no key is associated with index {@code i}
     */
    public void increaseKey(int i, Key key) {
        validateIndex(i);
        if (!contains(i)) throw new NoSuchElementException("index is not in the priority queue");
        if (keys[i].compareTo(key) == 0)
            throw new IllegalArgumentException("Calling increaseKey() with a key equal to the key in the priority queue");
        if (keys[i].compareTo(key) > 0)
            throw new IllegalArgumentException("Calling increaseKey() with a key strictly less than the key in the priority queue");
        keys[i] = key;
        sink(qp[i]);
    }

    /**
     * Remove the key associated with index {@code i}.
     *
     * @param  i the index of the key to remove
     * @throws IllegalArgumentException unless {@code 0 <= i < maxN}
     * @throws NoSuchElementException no key is associated with index {@code i}
     */
    public void delete(int i) {
        validateIndex(i);
        if (!contains(i)) throw new NoSuchElementException("index is not in the priority queue");
        int index = qp[i];
        exch(index, n--);
        swim(index);
        sink(index);
        keys[i] = null;
        qp[i] = -1;
    }

    // throw an IllegalArgumentException if i is an invalid index
    private void validateIndex(int i) {
        if (i < 0) throw new IllegalArgumentException("index is negative: " + i);
        if (i >= maxN) throw new IllegalArgumentException("index >= capacity: " + i);
    }

   /***************************************************************************
    * General helper functions.
    ***************************************************************************/
    private boolean greater(int i, int j) {
        return keys[pq[i]].compareTo(keys[pq[j]]) > 0;
    }

    private void exch(int i, int j) {
        int swap = pq[i];
        pq[i] = pq[j];
        pq[j] = swap;
        qp[pq[i]] = i;
        qp[pq[j]] = j;
    }


   /***************************************************************************
    * Heap helper functions.
    ***************************************************************************/
    private void swim(int k) {
        while (k > 1 && greater(k/2, k)) {
            exch(k, k/2);
            k = k/2;
        }
    }

    private void sink(int k) {
        while (2*k <= n) {
            int j = 2*k;
            if (j < n && greater(j, j+1)) j++;
            if (!greater(k, j)) break;
            exch(k, j);
            k = j;
        }
    }
  }
